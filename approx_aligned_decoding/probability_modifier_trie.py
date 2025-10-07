import random
from typing import Union, List, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream


def convert_tensor_to_list(t : Union[torch.Tensor, List[List[int]]]) -> List[List[int]]:
    if isinstance(t, torch.Tensor):
        return t.tolist()
    else:
        return t

class ProbabilityModifierTrieNode:
    def __init__(self, orig_probs: torch.Tensor, eos_tok_id: int,
                 parent: Optional["ProbabilityModifierTrieNode"], id_in_parent: Optional[int]):
        # INVARIANT: modified_probs.sum() * (1/probability_scaling_factor) = 1
        self.eos_tok_id = eos_tok_id
        self.orig_probs = orig_probs
        self.modified_probs = orig_probs.clone().detach().to(torch.float64)
        self.modified_probs = self.modified_probs / self.modified_probs.sum()
        self.probability_mass_remaining: float = 1.0
        self.children = {}
        self.hallucinated_children = []
        self.parent = None
        self.id_in_parent = None
        if parent:
            self.parent = parent
            self.id_in_parent = id_in_parent


    @classmethod
    def construct_root(cls, probs: torch.Tensor, sampled_toks: Optional[torch.Tensor], eos_tok_id: int,
                       hallucination_indices: Optional[List[Optional[int]]] = None):
        root = cls(probs[0, 0], eos_tok_id, None, None)
        if sampled_toks is None:
            return root

        num_generations = sampled_toks.shape[0]
        assert num_generations == probs.shape[0]

        generation_length = sampled_toks.shape[1]
        assert generation_length + 1 == probs.shape[1]  # Include root

        root.add_children(probs[:, 1:], sampled_toks, hallucination_indices)
        return root

    def add_children_at_idx(self, idx: int, probs: torch.Tensor, sampled_toks: Optional[torch.Tensor],
                            hallucination_indices: Optional[List[Optional[int]]]):
        if idx in self.children:
            root = self.children[idx]
        else:
            root = ProbabilityModifierTrieNode(probs[0, 0], self.eos_tok_id, self, idx)
            self.children[idx] = root

        if sampled_toks is not None:
            root.add_children(probs[:, 1:], sampled_toks, hallucination_indices)

        return root

    def get_cumulative_prob_mass_trace(self):
        parent_path = self.get_parent_path()
        # Begin at root
        current_node = self.get_nth_parent(len(parent_path))
        trace = []
        cumulative_prob_mass = 1.0
        for tok_id in parent_path:
            prob_of_tok_in_current_node = float(
                current_node.modified_probs[tok_id] / current_node.probability_mass_remaining)
            cumulative_prob_mass *= prob_of_tok_in_current_node
            trace.append(cumulative_prob_mass)
            current_node = current_node.children[tok_id]

        return trace

    def get_selection_prob_trace(self, gen_tokens: DecodedTokenStream):
        parent_path = self.get_parent_path()
        # Begin at root
        current_node = self.get_nth_parent(len(parent_path))
        trace = []
        for tok_id in gen_tokens.tokens:
            prob_of_tok_in_current_node = float(
                current_node.modified_probs[tok_id] / current_node.probability_mass_remaining)
            trace.append(prob_of_tok_in_current_node)
            current_node = current_node.children.get(tok_id)

        return trace

    def calc_probabilistic_backtracking(self, old_sel_prob_trace: List[float], gen_tokens: DecodedTokenStream) -> \
    Optional[Tuple[int, int]]:
        """
        IMPORTANT ASSUMPTION: between when old_sel_prob_trace was collected and calling this, only the current node
        (or its children) were marked as hallucinations.
        This allows us to skip calculating residuals on the token that we backtrack to,
         because any "excess probability" gets distributed evenly to all sibling branches
        :return: If we would have chosen a different path, the token index and token itself, otherwise None
        """
        current_node = self.get_nth_parent(len(self.get_parent_path()))
        assert current_node.parent is None  # At root

        for idx, (old_prob, sel_token) in enumerate(zip(old_sel_prob_trace, gen_tokens.tokens)):
            prob_of_tok_in_parent = current_node.modified_probs[
                                        sel_token].item() / current_node.probability_mass_remaining

            prob_ratio = min(1.0, prob_of_tok_in_parent / old_prob)

            if random.random() < prob_ratio:
                current_node = current_node.children.get(sel_token)
            else:
                # See assumption in docstring
                residuals = current_node.modified_probs.clone()
                residuals[sel_token] = 0
                assert residuals.sum() != 0
                residuals = residuals / residuals.sum()
                new_token = torch.multinomial(residuals, 1)[0].item()
                return idx, new_token

        return None

    def add_children(self, probs: torch.Tensor, sampled_toks: torch.Tensor, hallucination_indices: Optional[List[Optional[int]]] = None):
        num_generations = probs.shape[0]
        assert num_generations == sampled_toks.shape[0]

        generation_length = sampled_toks.shape[1]
        assert generation_length == probs.shape[1]

        node_at_level = [self] * num_generations
        run_this_generation = [True] * num_generations

        # Would be nice to do this with recursive calls, but Python has a limited stack frame :(
        for token_idx in range(generation_length):
            if not any(run_this_generation):
                break

            this_step = sampled_toks[:, token_idx].tolist()
            next_node_at_level = []

            for generation_idx in range(num_generations):
                if not run_this_generation[generation_idx]:
                    next_node_at_level.append(None)
                    continue

                self_node = node_at_level[generation_idx]

                tok = this_step[generation_idx]
                if tok == self.eos_tok_id:  # probs[generation_idx, token_idx] tells us what comes after EOS, it is therefore useless
                    run_this_generation[generation_idx] = False

                if tok not in self_node.children:
                    self_node.children[tok] = ProbabilityModifierTrieNode(probs[generation_idx, token_idx],
                                                                          self_node.eos_tok_id, self_node, tok)

                if hallucination_indices and hallucination_indices[generation_idx] == token_idx:
                    self_node.children[tok].mark_as_hallucination()

                next_node_at_level.append(self_node.children[tok])

            node_at_level = next_node_at_level

    def mark_child_as_hallucination(self, token_id: int):
        if token_id in self.hallucinated_children:
            return

        self.hallucinated_children.append(token_id)

        cumulative_reverse_probability = 1.0
        if token_id in self.children:
            self.children[token_id].probability_mass_remaining = 0.0

        node = self
        token_id_of_child = token_id
        while True:
            if node is None:
                break
            if node.probability_mass_remaining == 0.0:
                # This is possible if, in a single pass, all children of the node with nonzero probs are marked as hallu
                # and then some node with zero probs is also marked as a hallucination
                # Don't need to further propagate up because the node is useless at this point
                break

            prob_in_parent = (node.modified_probs[token_id_of_child].item() / node.probability_mass_remaining)
            if prob_in_parent > 1:
                assert prob_in_parent < 1 + 1e-5  # The floating point errors here can get surprisingly huge
                prob_in_parent = 1

            cumulative_reverse_probability *= prob_in_parent
            node.modified_probs[token_id_of_child] -= (
                    cumulative_reverse_probability * node.probability_mass_remaining)
            if node.modified_probs[token_id_of_child] < 1e-12:
                assert node.modified_probs[token_id_of_child] > -1e-8  # Just floating point error, safe to ignore
                node.modified_probs[token_id_of_child] = 0
            node.probability_mass_remaining -= (cumulative_reverse_probability * node.probability_mass_remaining)
            if node.probability_mass_remaining < 0:
                assert node.probability_mass_remaining > -1e-8
                node.probability_mass_remaining = 0

            if node.probability_mass_remaining < 1e-4:  # Floating point errors accumulate, catch them early
                node.probability_mass_remaining = node.modified_probs.sum().item()

            token_id_of_child = node.id_in_parent
            if not node.parent:
                break

            if token_id_of_child in node.parent.hallucinated_children:
                # print(repr(self))
                # This shouldn't happen except in very rare edge cases with the hallucination detector.
                # But if it does happen, it's pointless to continue up further since the probability of choosing this
                # path is zero anyways
                break
            node = node.parent

    def mark_as_hallucination(self):
        token_id_in_parent = self.id_in_parent
        if self.parent is None:
            return

        self.parent.mark_child_as_hallucination(token_id_in_parent)

    def get_probs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return (self.modified_probs.to(
            device) if device is not None else self.modified_probs) / self.probability_mass_remaining

    def get_orig_probs(self) -> torch.Tensor:
        return self.orig_probs

    def get_child(self, child_or_seq: Union[int, Sequence[int]]):
        if isinstance(child_or_seq, int):
            return self.children[child_or_seq]

        node = self
        for tok in child_or_seq:
            if tok not in node.children:
                raise IndexError
            node = node.children[tok]

        return node

    def get_parent_path(self):
        parent_path = []
        node = self
        while True:
            if node.parent is None:
                break
            parent_path.append(node.id_in_parent)
            node = node.parent

        return parent_path[::-1]

    def get_nth_parent(self, n):
        node = self
        for i in range(n):
            if node.parent is None:
                raise IndexError
            node = node.parent

        return node

    def is_hallucination(self):
        if self.parent is None:
            return False
        return self.id_in_parent in self.parent.hallucinated_children

    def get_parent_trace(self):
        parent_path = self.get_parent_path()
        root = self.get_nth_parent(len(parent_path))
        for tok in parent_path:
            yield root
            root = root.children[tok]
        yield root

    def full_prob_trace(self, tokenizer: Tokenizer, max_num_per_level: int = 5) -> List[List[Tuple[str, float]]]:
        result = []
        for i in self.get_parent_trace():
            probs, indices = i.get_probs("cpu").sort(0, True)
            num_nonzero_probs = (probs > 0).sum().item()
            result.append([(tokenizer.decode(indices[j].item()), probs[j].item()) for j in
                           range(min(num_nonzero_probs, max_num_per_level))])

        return result

    def __repr__(self):
        return f"ProbabilityModifierTrieNode(is_hallucination={self.is_hallucination()}, parent_path={self.get_parent_path()})"
