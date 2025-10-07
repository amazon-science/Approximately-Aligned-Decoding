import json
import pathlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, List, Tuple, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, LogitsProcessorList, TemperatureLogitsWarper, \
    TopKLogitsWarper, TopPLogitsWarper
from typing_extensions import Any

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream
from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector
from approx_aligned_decoding.model_with_history import ModelWithHistory
from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode
from approx_aligned_decoding.stopping_criteria.stopping_criteria import StoppingCriteria

_hallu_log_file: Optional[pathlib.Path] = None
_current_data_idx = None
_current_generation_num = None


def set_hallucination_logger(path: pathlib.Path):
    global _hallu_log_file
    _hallu_log_file = path


def set_current_data_idx(idx, gen_num):
    global _current_data_idx, _current_generation_num
    _current_data_idx = idx
    _current_generation_num = gen_num


def log_hallucination(left_context, right_context, generation, index_of_hallucination, index_prev_checked):
    global _hallu_log_file, _current_data_idx, _current_generation_num

    if _hallu_log_file is not None:
        with(open(_hallu_log_file, "a")) as file:
            json.dump({
                "data_idx": _current_data_idx,
                "generation_num": _current_generation_num,
                "left_context": left_context + generation[:index_prev_checked],
                "right_context": right_context,
                "generation": generation[index_prev_checked:],
                "index_of_hallucination": index_of_hallucination - index_prev_checked,
            }, file)
            file.write("\n")


def get_hallucination_indices(
        input_ids: List[DecodedTokenStream],
        tokenizer: PreTrainedTokenizer,
        hallucination_detector: HallucinationDetector,
        left_context: str,
        right_context: str,
        prev_tok_idx: int
) -> List[Optional[int]]:
    """
    Bridge between the hallucination detector interface and a more useful token-based interface
    :param left_context: Not included in input_ids- appended prior
    :param right_context: Not included in input_ids- appended after
    :param prev_tok_idx: If hallucination checking is expensive, only check hallus after this token index
    """
    outputs = []
    for generation_index, this_tok_stream in enumerate(input_ids):
        relevant_tok_ids = this_tok_stream.tokens
        relevant_decoded = this_tok_stream.decoded

        is_end = False
        if tokenizer.eos_token_id in relevant_tok_ids:
            is_end = True
            index_of_eos = relevant_tok_ids.index(tokenizer.eos_token_id)
            relevant_tok_ids = relevant_tok_ids[:index_of_eos]
            relevant_decoded = relevant_decoded[:index_of_eos]

        generation = "".join(relevant_decoded)

        index_prev_checked = (this_tok_stream.offsets[prev_tok_idx - 1] if prev_tok_idx != 0 else 0)

        detection_result = hallucination_detector.detect(
            left_context=left_context,
            right_context=right_context,
            generated_text=generation,
            is_end=is_end,
            index_previously_checked=index_prev_checked
        )

        if detection_result.index_of_hallucination is None:
            outputs.append(None)
        else:
            # Get token associated with the character index
            outputs.append(this_tok_stream.char_to_tok(detection_result.index_of_hallucination) - prev_tok_idx)
            log_hallucination(left_context, right_context, generation, detection_result.index_of_hallucination,
                              index_prev_checked)

    return outputs


@dataclass
class GenerationStats:
    num_backtracks: int
    total_generated_toks_model: int
    total_invocations_model: int
    total_generated_toks_speculative: Optional[int] = None
    total_invocations_speculative: Optional[int] = None
    limit_reached: bool = False


def generate(model: PreTrainedModel,
             input_ids: torch.Tensor,
             max_new_tokens: int,
             tokenizer: PreTrainedTokenizer,
             left_context: str,
             right_context: str,
             hallucination_detector: Optional[HallucinationDetector] = None,
             stopping_criteria: Optional[StoppingCriteria] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: float = 1,
             seed_func: Callable[[Any], Any] = lambda x: 0,
             backtracking_strategy: Optional[BacktrackingStrategy] = None,
             attention_mask: Optional[torch.Tensor] = None,  # So that we can **(tokenizer(...))
             token_type_ids: Optional[torch.Tensor] = None,  # Again so that we can **tokenizer
             print_probability_trace_info: bool = False,
             move_hallu_idx_forward: bool = False,
             max_total_invocations: Optional[int] = None,
             ) -> Tuple[Sequence[int], GenerationStats]:
    seed_func(0)

    warpers = LogitsProcessorList()
    if temperature != 1:
        assert temperature != 0
        warpers.append(TemperatureLogitsWarper(temperature=temperature))

    if top_k is not None:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=2))

    if top_p is not None:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=2))

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only batch size 1 supported right now"

    device_model = model.device
    num_existing_tokens = input_ids.shape[1]
    current_tok_idx = num_existing_tokens
    model.eval()

    gen_tokens = DecodedTokenStream(tokenizer)

    model_hist = ModelWithHistory(model, device_model, warper=warpers)

    # Catch up main model as if it has just generated the last token (but hasn't autoregressively predicted on it yet)
    if input_ids.shape[1] > 1:
        model_hist = model_hist(input_ids[:, :-1])[1][0]

    probability_modifer_trie_root = None
    current_probability_modifier_trie_pos = None
    stats = GenerationStats(0, 0, 0)
    last_token_id = None

    last_before_backtrack = None

    def backtrack_to(tok_idx, replacement_token):
        nonlocal gen_tokens, current_probability_modifier_trie_pos, last_token_id, current_tok_idx
        nonlocal model_hist, last_before_backtrack

        last_before_backtrack = gen_tokens

        if gen_tokens.tokens[tok_idx] == replacement_token:
            # There was a bug where we overcounted the generation ratio because if the replacement token
            # is the same as what is already there, we don't need to re-evaluate the LLM.
            # This corrects for that without changing the results of the experiments for same random seed.
            stats.total_generated_toks_model -= 1
            stats.total_invocations_model -= 1

        gen_tokens = gen_tokens.truncate(tok_idx)
        current_probability_modifier_trie_pos = probability_modifer_trie_root.get_child(gen_tokens.tokens)

        current_tok_idx = num_existing_tokens + tok_idx + 1
        gen_tokens = gen_tokens.add_toks([replacement_token])

        model_hist = model_hist.truncate(current_tok_idx - 1)

        last_token_id = replacement_token
        stats.num_backtracks += 1

    with (torch.inference_mode()):
        while current_tok_idx < num_existing_tokens + max_new_tokens:
            seed_func((current_tok_idx, stats.num_backtracks, 1))

            if last_token_id is None:
                model_inputs = input_ids[:, -1:].to(device_model)
            else:
                model_inputs = torch.tensor(last_token_id, device=device_model).view((1, 1))

            if max_total_invocations is not None:
                if stats.total_invocations_model >= max_total_invocations:
                    stats.limit_reached = True
                    return last_before_backtrack.tokens[:-1], stats  # Cut off last token (presumably containing error)

            model_outputs, [model_hist] = model_hist(model_inputs)
            main_probs = torch.softmax(model_outputs.logits, dim=-1)

            stats.total_invocations_model += 1
            stats.total_generated_toks_model += 1

            if probability_modifer_trie_root is None:
                probability_modifer_trie_root = ProbabilityModifierTrieNode.construct_root(probs=main_probs.cpu(),
                                                                                           sampled_toks=None,
                                                                                           eos_tok_id=model.config.eos_token_id)
                current_probability_modifier_trie_pos = probability_modifer_trie_root
            else:
                assert last_token_id is not None
                current_probability_modifier_trie_pos = \
                    current_probability_modifier_trie_pos.add_children_at_idx(idx=last_token_id,
                                                                              probs=main_probs.cpu(),
                                                                              sampled_toks=None,
                                                                              hallucination_indices=([None]))

            probability_at_depth = current_probability_modifier_trie_pos.get_probs(device_model)
            last_token_id = _sample_single_node_with_fallback(probability_at_depth,
                                                              fallback_probs=current_probability_modifier_trie_pos.get_orig_probs())
            gen_tokens = gen_tokens.add_toks([last_token_id])
            current_tok_idx += 1

            if hallucination_detector and backtracking_strategy and backtracking_strategy.do_whole_sample_hallu_check():
                seed_func((current_tok_idx, stats.num_backtracks, 2))
                while True:
                    backtracking_strategy.pre_sample_whole_generation_check(
                        probability_trie_node=current_probability_modifier_trie_pos, gen_tokens=gen_tokens)

                    hallu_index = get_hallucination_indices(input_ids=[gen_tokens],
                                                            tokenizer=tokenizer,
                                                            hallucination_detector=hallucination_detector,
                                                            left_context=left_context,
                                                            right_context=right_context,
                                                            prev_tok_idx=0)[0]

                    if hallu_index is not None:
                        if move_hallu_idx_forward:
                            hallu_index = len(gen_tokens.tokens) - 1

                        if print_probability_trace_info:
                            print("Hallucination detected")
                            print(tokenizer.decode(gen_tokens.tokens))
                            print(current_probability_modifier_trie_pos.full_prob_trace(tokenizer))

                        tok_id_of_hallu = gen_tokens.tokens[hallu_index]
                        probability_modifer_trie_root.get_child(
                            gen_tokens.tokens[:hallu_index]).mark_child_as_hallucination(tok_id_of_hallu)

                        if print_probability_trace_info:
                            print("Modified Probs:")
                            print(current_probability_modifier_trie_pos.full_prob_trace(tokenizer))

                    backtrack_result = backtracking_strategy.post_sample_whole_generation_check(
                        probability_trie_node=current_probability_modifier_trie_pos,
                        gen_tokens=gen_tokens,
                        hallu_index=hallu_index)

                    if backtrack_result is not None:
                        # The backtracker proposed a replacement token, we need to check that for hallucinations
                        # so keep looping
                        backtrack_idx, replacement_token = backtrack_result
                        backtrack_to(backtrack_idx, replacement_token)
                        continue
                    else:
                        # Backtracker is happy with the current situation, stop looping
                        break

            is_eos = (last_token_id == model.config.eos_token_id)

            decoded_text = "".join(gen_tokens.decoded)
            if stopping_criteria is not None and stopping_criteria.should_stop(left_context=left_context,
                                                                               right_context=right_context,
                                                                               generated=decoded_text):
                is_eos = True

            if is_eos:
                break

    if print_probability_trace_info:
        print("Final")
        print(current_probability_modifier_trie_pos.full_prob_trace(tokenizer))

    return gen_tokens.tokens, stats


USE_DEBUG_NON_SPECULATIVE_SAMPLING = False


def _debug_nonspeculative_sampling(
        main_model_probs,
        eos_id
):
    new_token = torch.multinomial(main_model_probs[0, 0], 1).item()
    return [new_token], new_token == eos_id, 0


def _sample_single_node_with_fallback(probs, fallback_probs):
    try:
        new_token = torch.multinomial(probs, 1)
    except RuntimeError:
        # This shouldn't really happen, but occasionally does due to floating point issues causing an inf or nan
        # Or top-k sampling leading to a fairly sparse set of options
        # If it does happen, use orig probs as a reasonable default
        # (sample from model instead of residuals or modified probs)
        new_token = torch.multinomial(fallback_probs, 1)

    return new_token[0].item()


def _speculative_sampling_one_round(probability_trie: ProbabilityModifierTrieNode,
                                    assistant_input_ids: torch.Tensor,
                                    assistant_probs: torch.Tensor,
                                    eos_id: int):
    num_threads = assistant_input_ids.shape[0]
    max_depth = assistant_input_ids.shape[1]

    active_threads = torch.ones(num_threads, dtype=torch.bool, device=assistant_input_ids.device)
    idx_of_some_active = 0
    depth = 0
    final_new_tokens = []

    current_prob_trie_node = probability_trie

    is_eos = False
    while active_threads.sum() > 0 and depth < max_depth:
        assistant_input_ids_at_depth = assistant_input_ids[:, depth]
        assistant_probs_at_depth = assistant_probs[idx_of_some_active, depth]

        main_model_probs_at_depth = current_prob_trie_node.get_probs(assistant_probs.device)
        orig_main_model_probs_at_depth = current_prob_trie_node.get_orig_probs().to(assistant_probs.device)

        new_token = None

        prob_ratios = main_model_probs_at_depth / assistant_probs_at_depth

        for thread in range(num_threads):
            if active_threads[thread]:
                assistant_choice = assistant_input_ids_at_depth[thread]
                r = random.random()
                if r < prob_ratios[assistant_choice]:
                    new_token = assistant_input_ids_at_depth[thread].item()
                    break
                else:
                    main_model_probs_at_depth = torch.clamp(main_model_probs_at_depth - assistant_probs_at_depth, min=0)
                    if main_model_probs_at_depth.sum() == 0:
                        # More likely to happen with small top-k or any top-p sampling
                        # At this point just choose a reasonable default of sampling directly from main model
                        new_token = torch.multinomial(orig_main_model_probs_at_depth, 1)[0].item()
                        break
                    main_model_probs_at_depth = main_model_probs_at_depth / main_model_probs_at_depth.sum()
                    prob_ratios = main_model_probs_at_depth / assistant_probs_at_depth

        if new_token is None:  # None of the speculative sequences match, at least get the "free" token at the end
            assert not main_model_probs_at_depth.sum() == 0

            new_token = _sample_single_node_with_fallback(main_model_probs_at_depth, orig_main_model_probs_at_depth)
            final_new_tokens.append(new_token)
            break

        final_new_tokens.append(new_token)

        if new_token == eos_id:
            is_eos = True
            break

        active_threads &= (assistant_input_ids_at_depth == new_token)
        active_threads_indices = torch.argwhere(active_threads)
        if active_threads_indices.shape[0] > 0:
            idx_of_some_active = active_threads_indices[0][0]
        else:
            break

        # This new_token might not exist in the probability trie (if the fallback option is used for sum probs = 0)
        # So this needs to go after the check of active threads
        current_prob_trie_node = current_prob_trie_node.get_child(new_token)
        depth += 1

    if depth == max_depth:
        # Sample from main model
        # main_model_probs should reflect the conditional probabilities at index_of_some_active
        new_token = _sample_single_node_with_fallback(current_prob_trie_node.get_probs(),
                                                      current_prob_trie_node.get_orig_probs())
        final_new_tokens.append(new_token)
        if new_token == eos_id:
            is_eos = True

    return final_new_tokens, is_eos, idx_of_some_active


def _speculative_sampling(
        assistant_input_ids,
        assistant_probs,
        main_model_probs,
        eos_id,
        hallucination_indices: Optional[List[Optional[int]]] = None,
        num_final_token_iters: int = 1,
        hallucination_index_finder: Callable[[List[int]], Optional[int]] = lambda x: None,
        probability_trie: Optional[ProbabilityModifierTrieNode] = None
):
    """
    Applies sampling as in the tree-based speculative decoding paper: https://arxiv.org/pdf/2305.09781

    """

    num_threads = assistant_input_ids.shape[0]
    max_depth = assistant_input_ids.shape[1]
    assert assistant_probs.shape[0] == num_threads
    assert main_model_probs.shape[0] == num_threads
    assert assistant_probs.shape[1] == max_depth
    assert main_model_probs.shape[1] == max_depth + 1
    assert assistant_probs.shape[2] == main_model_probs.shape[2]

    if probability_trie is None:
        probability_trie = ProbabilityModifierTrieNode.construct_root(probs=main_model_probs.cpu(),
                                                                      sampled_toks=assistant_input_ids.cpu(),
                                                                      eos_tok_id=eos_id,
                                                                      hallucination_indices=hallucination_indices)

    for i in range(num_final_token_iters - 1):
        new_tokens, is_eos, gen_idx = _speculative_sampling_one_round(probability_trie=probability_trie,
                                                                      assistant_probs=assistant_probs,
                                                                      assistant_input_ids=assistant_input_ids,
                                                                      eos_id=eos_id)
        hallu_index = hallucination_index_finder(new_tokens)
        if hallu_index is not None:
            probability_trie.get_child(new_tokens[:-1]).mark_child_as_hallucination(new_tokens[-1])
        else:
            return new_tokens, is_eos, gen_idx

    # Reached max retries, allow leakage
    return _speculative_sampling_one_round(probability_trie=probability_trie,
                                           assistant_probs=assistant_probs,
                                           assistant_input_ids=assistant_input_ids,
                                           eos_id=eos_id)


def generate_with_asst(model: PreTrainedModel,
                       assistant_model: PreTrainedModel,
                       input_ids: torch.Tensor,
                       max_new_tokens: int,
                       num_speculative_sequences: int, spec_sequence_length: int,
                       tokenizer: PreTrainedTokenizer,
                       left_context: str,
                       right_context: str,
                       hallucination_detector: Optional[HallucinationDetector] = None,
                       attention_mask: Optional[torch.Tensor] = None,  # So that we can **(tokenizer(...))
                       token_type_ids: Optional[torch.Tensor] = None,  # Again so that we can **tokenizer
                       stopping_criteria: Optional[StoppingCriteria] = None,
                       top_k: Optional[int] = None,
                       top_p: Optional[float] = None,
                       temperature: float = 1,
                       seed_func: Callable[[Any], Any] = lambda x: 0,
                       backtracking_strategy: Optional[BacktrackingStrategy] = None,
                       move_hallu_idx_forward: bool = False,
                       ) -> Tuple[Sequence[int], GenerationStats]:
    """
    NOTE: This mode is more experimental compared to generate
    It is mostly a proof of concept to show that our method can coexist with speculative decoding,
    but it does not increase performance due to an inefficient implementation of batching, no tree attention, etc.
    """
    seed_func(0)

    warpers = LogitsProcessorList()
    if temperature != 1:
        assert temperature != 0
        warpers.append(TemperatureLogitsWarper(temperature=temperature))

    if top_k is not None:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=2))

    if top_p is not None:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=2))

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only batch size 1 supported right now"

    device_model = model.device
    device_asst = assistant_model.device
    num_existing_tokens = input_ids.shape[1]
    current_tok_idx = num_existing_tokens
    model.eval()

    gen_tokens = DecodedTokenStream(tokenizer)

    model_hist = ModelWithHistory(model, device_model, warper=warpers)

    # Catch up main model as if it has just generated the last token (but hasn't autoregressively predicted on it yet)
    if input_ids.shape[1] > 1:
        model_hist = model_hist(input_ids[:, :-1])[1][0]

    assistant_hist = ModelWithHistory(assistant_model, device_asst, warper=warpers)
    assistant_model_inputs = input_ids.to(device_asst)
    probability_modifer_trie_root = None
    current_probability_modifier_trie_pos = None
    stats = GenerationStats(0, 0, 0, 0, 0)

    def backtrack_to(tok_idx, replacement_token):
        nonlocal gen_tokens, current_probability_modifier_trie_pos, last_token_id, current_tok_idx
        nonlocal model_hist, assistant_hist, assistant_model_inputs, stats

        gen_tokens = gen_tokens.truncate(tok_idx)
        current_probability_modifier_trie_pos = probability_modifer_trie_root.get_child(gen_tokens.tokens)

        current_tok_idx = num_existing_tokens + tok_idx + 1
        gen_tokens = gen_tokens.add_toks([replacement_token])

        model_hist = model_hist.truncate(current_tok_idx - 1)
        if len(assistant_hist) == current_tok_idx - 2:
            # Special case: Generated extra "free" token, and that token is a hallucination
            # Assistant model will be one behind the main model, and won't have the hallucinated token in its history
            # So we just change up its inputs in the next round
            assert assistant_model_inputs.shape[0] == 1
            assert assistant_model_inputs.shape[1] == 2
            assistant_model_inputs = assistant_model_inputs.clone()
            assistant_model_inputs[0, 1] = replacement_token
        else:
            assistant_hist = assistant_hist.truncate(current_tok_idx - 1)
            assistant_model_inputs = torch.tensor(replacement_token, device=device_asst).view((1, 1))
        last_token_id = replacement_token
        stats.num_backtracks += 1

    with (torch.inference_mode()):
        while current_tok_idx < num_existing_tokens + max_new_tokens:
            seed_func((current_tok_idx, stats.num_backtracks, 1))

            num_new_tokens = current_tok_idx - num_existing_tokens
            spec_sequence_length_this_round = min(spec_sequence_length, max_new_tokens - num_new_tokens)
            raw_spec_outputs, spec_outputs_models, num_spec_tokens = assistant_hist.generate(
                input_ids=assistant_model_inputs,
                max_new_tokens=spec_sequence_length_this_round,
                num_return_sequences=num_speculative_sequences)

            stats.total_invocations_speculative += spec_sequence_length_this_round
            stats.total_generated_toks_speculative += (spec_sequence_length_this_round * num_speculative_sequences)

            num_new_spec_tokens = raw_spec_outputs.sequences.shape[1]
            new_spec_output_logits = raw_spec_outputs.logits[:, -num_new_spec_tokens:]
            new_spec_output_probs = torch.softmax(new_spec_output_logits, dim=-1)

            # Include the last input of assistant model in addition to all of its outputs
            main_input = torch.cat(
                [assistant_model_inputs[:, -1].tile((num_speculative_sequences, 1)), raw_spec_outputs.sequences], dim=1)
            main_outputs, main_outputs_models = model_hist(main_input.to(device_model))
            main_probs = torch.softmax(main_outputs.logits, dim=-1).to(device_asst)

            stats.total_invocations_model += 1
            stats.total_generated_toks_model += (num_speculative_sequences * num_new_spec_tokens)

            seq_to_list = raw_spec_outputs.sequences.tolist()
            gen_tokens_threads = [gen_tokens.add_toks(seq) for seq in seq_to_list]

            # We have the main model probabilities at the root (or for the token that wasn't speculated in the previous round)
            # Add that to the tree in preparation for the actually new tokens
            if probability_modifer_trie_root is None:
                probability_modifer_trie_root = ProbabilityModifierTrieNode.construct_root(
                    probs=main_probs[:, 0:1].cpu(),
                    sampled_toks=None,
                    eos_tok_id=model.config.eos_token_id)
                current_probability_modifier_trie_pos = probability_modifer_trie_root
            else:
                assert last_token_id is not None
                current_probability_modifier_trie_pos = \
                    current_probability_modifier_trie_pos.add_children_at_idx(idx=last_token_id,
                                                                              probs=main_probs[:, 0:1].cpu(),
                                                                              sampled_toks=None,
                                                                              hallucination_indices=([
                                                                                                         None] * num_speculative_sequences))

            if hallucination_detector:
                if backtracking_strategy:
                    backtracking_strategy.pre_speculative_detector(
                        probability_trie_node=current_probability_modifier_trie_pos,
                        speculative_probs=new_spec_output_probs,
                        main_probs=main_probs,
                        sampled_toks=raw_spec_outputs.sequences,
                        gen_tokens=gen_tokens)

                hallucination_indices = get_hallucination_indices(
                    input_ids=gen_tokens_threads,
                    tokenizer=tokenizer,
                    left_context=left_context,
                    right_context=right_context,
                    hallucination_detector=hallucination_detector,
                    prev_tok_idx=current_tok_idx - num_existing_tokens
                )
                if move_hallu_idx_forward:
                    hallucination_indices = [len(gtt.tokens) - 1 if x is not None else x for x, gtt in
                                             zip(hallucination_indices, gen_tokens_threads)]

                seed_func((current_tok_idx, stats.num_backtracks, 2))

                current_probability_modifier_trie_pos.add_children(probs=main_probs[:, 1:].cpu(),
                                                                   sampled_toks=raw_spec_outputs.sequences.cpu(),
                                                                   hallucination_indices=hallucination_indices)

                if backtracking_strategy:
                    backtrack_result = backtracking_strategy.post_speculative_detector(
                        probability_trie_node=current_probability_modifier_trie_pos,
                        speculative_probs=new_spec_output_probs,
                        main_probs=main_probs,
                        sampled_toks=raw_spec_outputs.sequences,
                        gen_tokens=gen_tokens,
                        hallu_indices=hallucination_indices)
                    if backtrack_result is not None:
                        backtrack_idx, alternate_token = backtrack_result

                        if backtrack_idx == 0:
                            # Small hack to handle the fact that models don't start with the initial probabilities
                            # Shouldn't change the result, as the initial probs aren't dependent on anything
                            # in the generated text
                            model_hist = main_outputs_models[0]
                            assistant_hist = spec_outputs_models[0]

                        backtrack_to(backtrack_idx, alternate_token)
                        continue

                new_tokens, is_eos, gen_idx = _speculative_sampling_one_round(
                    probability_trie=current_probability_modifier_trie_pos,
                    assistant_probs=new_spec_output_probs,
                    assistant_input_ids=raw_spec_outputs.sequences,
                    eos_id=model.config.eos_token_id)

                current_probability_modifier_trie_pos = current_probability_modifier_trie_pos.get_child(new_tokens[:-1])
            else:
                if USE_DEBUG_NON_SPECULATIVE_SAMPLING:
                    new_tokens, is_eos, gen_idx = _debug_nonspeculative_sampling(
                        main_model_probs=main_probs,
                        eos_id=model.config.eos_token_id
                    )
                else:
                    seed_func((current_tok_idx, stats.num_backtracks, 2))
                    new_tokens, is_eos, gen_idx = _speculative_sampling(
                        assistant_input_ids=raw_spec_outputs.sequences,
                        assistant_probs=new_spec_output_probs,
                        main_model_probs=main_probs,
                        eos_id=model.config.eos_token_id,
                    )

            last_token_id = new_tokens[-1]
            seed_func((current_tok_idx, stats.num_backtracks, 3))
            current_tok_idx += len(new_tokens)

            if len(new_tokens) == num_new_spec_tokens + 1:
                # Reached end of spec sequence and went on to generate the extra "free" token
                assistant_hist = spec_outputs_models[gen_idx]
                assistant_model_inputs = torch.tensor(new_tokens[-2:], device=device_asst).view((1, 2))
                model_hist = main_outputs_models[gen_idx]
                gen_tokens = gen_tokens_threads[gen_idx].add_toks([new_tokens[-1]])
            else:
                # Main model diverged somewhere
                # Go to just before the divergence point
                assistant_hist = spec_outputs_models[gen_idx].truncate(current_tok_idx - 1)
                assistant_model_inputs = torch.tensor(new_tokens[-1], device=device_asst).view((1, 1))
                model_hist = main_outputs_models[gen_idx].truncate(current_tok_idx - 1)
                gen_tokens = gen_tokens_threads[gen_idx].truncate(current_tok_idx - num_existing_tokens - 1).add_toks(
                    [new_tokens[-1]])

            if backtracking_strategy is not None and backtracking_strategy.do_whole_sample_hallu_check():

                while True:
                    backtracking_strategy.pre_sample_whole_generation_check(
                        probability_trie_node=current_probability_modifier_trie_pos,
                        gen_tokens=gen_tokens,
                    )

                    # Do a final detection on what has been generated and backtrack if necessary
                    hallu_index = get_hallucination_indices(input_ids=[gen_tokens],
                                                            tokenizer=tokenizer,
                                                            hallucination_detector=hallucination_detector,
                                                            left_context=left_context,
                                                            right_context=right_context,
                                                            prev_tok_idx=0)[0]

                    if hallu_index is not None:
                        if move_hallu_idx_forward:
                            hallu_index = len(gen_tokens.tokens) - 1

                        tok_id_of_hallu = gen_tokens.tokens[hallu_index]
                        probability_modifer_trie_root.get_child(
                            gen_tokens.tokens[:hallu_index]).mark_child_as_hallucination(tok_id_of_hallu)

                    backtrack_result = backtracking_strategy.post_sample_whole_generation_check(
                        probability_trie_node=current_probability_modifier_trie_pos,
                        gen_tokens=gen_tokens,
                        hallu_index=hallu_index
                    )

                    if backtrack_result is not None:
                        backtrack_idx, replacement_token = backtrack_result
                        backtrack_to(backtrack_idx, replacement_token)
                        # The backtracker proposed a replacement token, we need to check that for hallucinations
                        # so keep looping
                    else:
                        # Backtracker is happy with the current situation, stop looping
                        break

            decoded_text = "".join(gen_tokens.decoded)
            if stopping_criteria is not None and stopping_criteria.should_stop(left_context=left_context,
                                                                               right_context=right_context,
                                                                               generated=decoded_text):
                is_eos = True

            if is_eos:
                break

    return gen_tokens.tokens, stats
