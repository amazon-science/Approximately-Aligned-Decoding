from typing import Optional, Tuple, List

import torch

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream
from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode


class FastGADBacktrackingStrategy(BacktrackingStrategy):
    """
    Our method for backtracking during generation- probabilistically backtracks according to the amount of
    probability mass lost by the detected hallucination
    """
    def __init__(self):
        self.probability_trace = None

    def reset_state(self):
        self.probability_trace = None

    def pre_speculative_detector(self, probability_trie_node: ProbabilityModifierTrieNode, sampled_toks: torch.Tensor,
                                 speculative_probs: torch.Tensor, main_probs: torch.Tensor,
                                 gen_tokens: DecodedTokenStream):
        self.probability_trace = probability_trie_node.get_selection_prob_trace(gen_tokens)

    def post_speculative_detector(self, probability_trie_node: ProbabilityModifierTrieNode, sampled_toks: torch.Tensor,
                                  speculative_probs: torch.Tensor, main_probs: torch.Tensor,
                                  hallu_indices: List[Optional[int]], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:
        if any(h is not None for h in hallu_indices):
            return probability_trie_node.calc_probabilistic_backtracking(self.probability_trace, gen_tokens)

    def pre_sample_whole_generation_check(self, probability_trie_node: ProbabilityModifierTrieNode,
                                          gen_tokens: DecodedTokenStream):

        self.probability_trace = probability_trie_node.get_selection_prob_trace(gen_tokens)

    def post_sample_whole_generation_check(self, probability_trie_node: ProbabilityModifierTrieNode,
                                           hallu_index: Optional[int], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:

        if hallu_index is not None:
            return probability_trie_node.calc_probabilistic_backtracking(self.probability_trace, gen_tokens)

    def do_whole_sample_hallu_check(self) -> bool:
        return True
