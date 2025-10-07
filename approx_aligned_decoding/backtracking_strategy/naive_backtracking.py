from typing import Optional, Tuple

import torch

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream
from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode


class NaiveBacktrackingStrategy(BacktrackingStrategy):
    """
    Naive constrained generation: requires --move-hallu-idx-forward because we assume the hallu index is
    probability trie depth + 1
    Slightly more intelligent than pure constrained generation because it will backtrack when the probability becomes 0
    (possible when top-p or top-k sampling is used)
    """
    def post_sample_whole_generation_check(self,
                                           probability_trie_node: ProbabilityModifierTrieNode,
                                           hallu_index: Optional[int],
                                           gen_tokens: DecodedTokenStream) -> Optional[Tuple[int, int]]:
        if hallu_index is None:
            return None

        if hallu_index == 0:
            assert len(probability_trie_node.get_parent_path()) == 0
            return 0, torch.multinomial(probability_trie_node.get_probs(), num_samples=1)[0].item()

        # Hallu index is depth of trie + 1
        assert hallu_index == len(probability_trie_node.get_parent_path()), "Probably requires --move-hallu-idx-forward"
        node = probability_trie_node
        current_depth = hallu_index - 1
        while node.parent is not None:
            assert node.id_in_parent == gen_tokens.tokens[current_depth]
            if node.probability_mass_remaining == 0:
                node = node.parent
                current_depth -= 1
            else:
                return current_depth, node.id_in_parent

        # Reached root when backtracking- just choose some other token that hasn't been seen
        return 0, torch.multinomial(node.get_probs(), num_samples=1)[0].item()

    def do_whole_sample_hallu_check(self) -> bool:
        return True
