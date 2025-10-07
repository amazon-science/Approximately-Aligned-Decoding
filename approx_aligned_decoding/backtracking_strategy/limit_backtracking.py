from typing import Optional, Tuple, List

import torch

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream
from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode


class LimitBacktracking(BacktrackingStrategy):
    """
    Combine with other backtracking strategies to ensure that backtracking doesn't enter an infinite loop
    by limiting the total number of backtracks allowed
    """

    def __init__(self, inner: BacktrackingStrategy, limit: int, silent: bool = False):
        self.inner = inner
        self.limit = limit
        self.seen = 0
        self.silent = silent
        self.has_printed_limit_reached = False

    def reset_state(self):
        self.seen = 0
        self.has_printed_limit_reached = False
        self.inner.reset_state()

    def limit_reached(self):
        if not self.has_printed_limit_reached:
            if not self.silent:
                print(f"Backtracking limit of {self.limit} reached! Rest of generation doesn't use backtracking")
            self.has_printed_limit_reached = True

    def pre_speculative_detector(self, probability_trie_node: ProbabilityModifierTrieNode, sampled_toks: torch.Tensor,
                                 speculative_probs: torch.Tensor, main_probs: torch.Tensor,
                                 gen_tokens: DecodedTokenStream):
        if self.seen >= self.limit:
            self.limit_reached()
            return
        self.inner.pre_speculative_detector(probability_trie_node, sampled_toks, speculative_probs, main_probs,
                                            gen_tokens)

    def post_speculative_detector(self, probability_trie_node: ProbabilityModifierTrieNode, sampled_toks: torch.Tensor,
                                  speculative_probs: torch.Tensor, main_probs: torch.Tensor,
                                  hallu_indices: List[Optional[int]], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:
        if self.seen >= self.limit:
            return None
        result = self.inner.post_speculative_detector(probability_trie_node, sampled_toks, speculative_probs,
                                                      main_probs,
                                                      hallu_indices, gen_tokens)
        if result is not None:
            self.seen += 1

        return result

    def pre_sample_whole_generation_check(self, probability_trie_node: ProbabilityModifierTrieNode,
                                          gen_tokens: DecodedTokenStream):
        if self.seen >= self.limit:
            self.limit_reached()
            return
        self.inner.pre_sample_whole_generation_check(probability_trie_node, gen_tokens)

    def post_sample_whole_generation_check(self, probability_trie_node: ProbabilityModifierTrieNode,
                                           hallu_index: Optional[int], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:
        if self.seen >= self.limit:
            return None

        result = self.inner.post_sample_whole_generation_check(probability_trie_node, hallu_index, gen_tokens)

        if result is not None:
            self.seen += 1

        return result

    def do_whole_sample_hallu_check(self) -> bool:
        return (self.seen < self.limit) and self.inner.do_whole_sample_hallu_check()
