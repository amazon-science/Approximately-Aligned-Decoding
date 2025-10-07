import json
import os
import traceback
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, Callable

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing_extensions import Any

from approx_aligned_decoding.backtracking_strategy.fast_gad_backtracking import FastGADBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.grammar_aligned_decoding_backtracking import \
    GrammarAlignedDecodingBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.limit_backtracking import LimitBacktracking
from approx_aligned_decoding.backtracking_strategy.naive_backtracking import NaiveBacktrackingStrategy
from approx_aligned_decoding.generator import generate_with_asst, set_hallucination_logger, \
    set_current_data_idx, \
    generate
from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector
from approx_aligned_decoding.stopping_criteria.stopping_criteria import StoppingCriteria
from approx_aligned_decoding.utils import set_seed


def wrap_ds(model: PreTrainedModel):
    try:
        import deepspeed
        print("Using Deepspeed")
        return deepspeed.init_inference(model, dtype=torch.half, replace_with_kernel_inject=True)
    except ImportError:
        print("Deepspeed not found! Using HF model.")
        return model


def setup_env_for_generation():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def add_standard_args(parser: ArgumentParser):
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--assistant-model', type=str, required=False)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--device-model', type=str, required=True)
    parser.add_argument('--device-asst', type=str, required=False)
    parser.add_argument('--use-detector', action='store_true')
    parser.add_argument('--num-speculative-sequences', type=int, default=10)
    parser.add_argument('--spec-sequence-length', type=int, default=5)
    parser.add_argument('--attempts-final-tok', type=int, default=1)
    parser.add_argument('--use-ds', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backtrack-strategy", type=str, default=None)
    parser.add_argument("--backtrack-limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--move-hallu-idx-forward", action='store_true')


@dataclass
class GenerationInput:
    task_id: str
    prompt: str
    left_context: str
    right_context: str


@dataclass
class GenerationOutput:
    task_id: str
    completion: str
    solution: str


def do_generation_dataset(args: Any, ignore_existing: bool,
                          dataset: Iterator[GenerationInput],
                          dataset_len: int,
                          output_processor: Callable[[GenerationOutput], Dict],
                          detector: HallucinationDetector,
                          stopping_criteria: StoppingCriteria,
                          max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device_model,
                                                 quantization_config=quant_config)

    assistant_model = None
    if args.assistant_model is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(args.assistant_model, device_map=args.device_asst,
                                                               quantization_config=quant_config)

    if args.use_ds:
        model = wrap_ds(model)
        if assistant_model is not None:
            assistant_model = wrap_ds(assistant_model)

    file_mode = "w"
    existing_num_samples = Counter()

    if os.path.exists(args.output) and not ignore_existing:
        with open(args.output, "r") as f:
            for line in f:
                data = json.loads(line)
                existing_num_samples[data["task_id"]] += 1

        file_mode = "a"

    if args.use_detector:
        output_without_extension = Path(args.output).with_suffix("")
        hallu_log = output_without_extension.parent / (output_without_extension.name + "_hallucinations.jsonl")
        set_hallucination_logger(hallu_log)

    backtracking_strategy = None

    if args.backtrack_strategy is not None:
        strategy_type: str = args.backtrack_strategy
        strategy_type = strategy_type.lower()

        if strategy_type == "naive":
            backtracking_strategy = NaiveBacktrackingStrategy()
            if not args.move_hallu_idx_forward:
                print("WARNING: Using constrained generation without move-hallu-idx-forward")
        elif strategy_type == "gad":
            backtracking_strategy = GrammarAlignedDecodingBacktrackingStrategy()
        elif strategy_type == "fastgad":
            backtracking_strategy = FastGADBacktrackingStrategy()
        else:
            raise Exception("Backtrack Strategy must be Naive, GAD, or FastGAD")

    if args.backtrack_limit is not None:
        backtracking_strategy = LimitBacktracking(backtracking_strategy, args.backtrack_limit)

    with open(args.output, file_mode) as output_file:
        for input_val in tqdm(dataset, total=dataset_len):
            for sample_num in range(existing_num_samples[input_val.task_id], args.num_samples):
                try:
                    set_current_data_idx(input_val.task_id, sample_num)

                    if backtracking_strategy is not None:
                        backtracking_strategy.reset_state()

                    inputs = tokenizer(input_val.prompt, return_tensors="pt").to(model.device)

                    if assistant_model is not None:
                        raw_output, stats = generate_with_asst(model, assistant_model, max_new_tokens=max_new_tokens,
                                                               num_speculative_sequences=args.num_speculative_sequences,
                                                               spec_sequence_length=args.spec_sequence_length,
                                                               hallucination_detector=detector if args.use_detector else None,
                                                               left_context=input_val.left_context,
                                                               right_context=input_val.right_context,
                                                               tokenizer=tokenizer,
                                                               stopping_criteria=stopping_criteria,
                                                               top_p=args.top_p, temperature=args.temperature,
                                                               seed_func=lambda x: set_seed(
                                                                   (args.seed, input_val.task_id, sample_num, x)),
                                                               backtracking_strategy=backtracking_strategy,
                                                               **inputs)
                    else:
                        raw_output, stats = generate(model=model,
                                                     max_new_tokens=max_new_tokens,
                                                     hallucination_detector=detector if args.use_detector else None,
                                                     left_context=input_val.left_context,
                                                     right_context=input_val.right_context,
                                                     tokenizer=tokenizer,
                                                     stopping_criteria=stopping_criteria,
                                                     top_p=args.top_p, temperature=args.temperature,
                                                     seed_func=lambda x: set_seed(
                                                         (args.seed, input_val.task_id, sample_num, x)),
                                                     backtracking_strategy=backtracking_strategy,
                                                     move_hallu_idx_forward=args.move_hallu_idx_forward,
                                                     **inputs)

                    generated = tokenizer.decode(raw_output)
                    result = output_processor(GenerationOutput(task_id=input_val.task_id,
                                                               completion=generated,
                                                               solution=input_val.prompt + generated))
                    result.update(asdict(stats))
                    result.update({"num_tokens": len(raw_output)})
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    err_path = Path(args.output).with_suffix(".err")
                    with open(err_path, "a") as err_file:
                        traceback.print_exc(file=err_file)
                    traceback.print_exc()
