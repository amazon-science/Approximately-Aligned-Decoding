from typing import Optional, Tuple, List

import torch

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream
from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode


def resample_at_root(node: ProbabilityModifierTrieNode):
    return torch.multinomial(node.get_nth_parent(len(node.get_parent_path())).get_probs(), num_samples=1)[0].item()


class GrammarAlignedDecodingBacktrackingStrategy(BacktrackingStrategy):
    """
    Grammar aligned decoding/ASAp: https://arxiv.org/abs/2405.21047
    Resamples at root when a negative output is detected (after adjusting the probabilities)
    """
    def do_whole_sample_hallu_check(self) -> bool:
        return True

    def post_speculative_detector(self, probability_trie_node: ProbabilityModifierTrieNode, sampled_toks: torch.Tensor,
                                  speculative_probs: torch.Tensor, main_probs: torch.Tensor,
                                  hallu_indices: List[Optional[int]], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:
        if any(h is not None for h in hallu_indices):
            return (0, resample_at_root(probability_trie_node))

    def post_sample_whole_generation_check(self, probability_trie_node: ProbabilityModifierTrieNode,
                                           hallu_index: Optional[int], gen_tokens: DecodedTokenStream) -> Optional[
        Tuple[int, int]]:
        if hallu_index is not None:
            return (0, resample_at_root(probability_trie_node))
