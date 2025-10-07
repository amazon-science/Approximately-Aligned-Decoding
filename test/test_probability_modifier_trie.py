import unittest
from collections import Counter

import numpy as np
from torch import tensor

from approx_aligned_decoding.probability_modifier_trie import ProbabilityModifierTrieNode


class TestProbabilityModifierTrie(unittest.TestCase):
    def test_example_no_hallucinations(self):
        tok_ids = [
            [0, 0],
            [0, 1],
            [1, 1]
        ]
        probs = [
            [[.5, .5], [.4, .6], [.99, .01]],
            [[.5, .5], [.4, .6], [.3, .7]],
            [[.5, .5], [.8, .2], [.5, .5]],
        ]
        trie = ProbabilityModifierTrieNode.construct_root(tensor(probs), tensor(tok_ids), 1)

        np.testing.assert_almost_equal(trie.get_probs().tolist(), [.5, .5])
        self.assertTrue(0 in trie.children)
        self.assertTrue(0 in trie.children[0].children)
        np.testing.assert_almost_equal(trie.children[0].get_probs().tolist(), [.4, .6])
        self.assertEqual(trie.children[0].children[0].children, {})
        np.testing.assert_almost_equal(trie.children[0].children[0].get_probs().tolist(), [.99, 0.01])


    def test_example_one_hallucination(self):
        tok_ids = [
            [0, 2, 0],
            [1, 2, 0],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 0],
            [1, 1, 1],
        ]
        probs = [
            [[0.5, 0.4, 0.1]] * 4,
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.5, 0], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]],
        ]

        hallucination_ids = [None, None, None, None, None, 2]

        trie = ProbabilityModifierTrieNode.construct_root(tensor(probs), tensor(tok_ids), 2, hallucination_ids)

        raw_probs = np.array([0.3, .6 - .6, 0.1])
        np.testing.assert_almost_equal(trie.get_child([1, 1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.2, 0.5 - (.6 * .5), 0.3])
        np.testing.assert_almost_equal(trie.get_child([1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4 - (.6 * .5 * .4), 0.1])
        np.testing.assert_almost_equal(trie.get_probs().tolist(), raw_probs / raw_probs.sum())

    def test_example_two_hallucinations(self):
        tok_ids = [
            [0, 2, 0],
            [1, 2, 0],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 0],
            [1, 1, 1],
        ]
        probs = [
            [[0.5, 0.4, 0.1]] * 4,
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]],
        ]

        hallucination_ids = [None, 1, None, None, None, 2]

        trie = ProbabilityModifierTrieNode.construct_root(tensor(probs), tensor(tok_ids), 2, hallucination_ids)

        raw_probs = np.array([0.3, .6 - .6, 0.1])
        np.testing.assert_almost_equal(trie.get_child([1, 1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.2, 0.5 - (.6 * .5), 0.3 - 0.3])
        np.testing.assert_almost_equal(trie.get_child([1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4 - (.6 * .5 * .4) - (.3 * .4), 0.1])
        np.testing.assert_almost_equal(trie.get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4, 0.1])
        np.testing.assert_almost_equal(trie.get_child([0]).get_probs().tolist(), raw_probs / raw_probs.sum())

    def test_parent_hallucination_parent_first(self):
        tok_ids = [
            [0, 2, 0],
            [1, 2, 0],
            [1, 2, 1],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 0],
            [1, 1, 1],
        ]

        probs = [
            [[0.5, 0.4, 0.1]] * 4,
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]],
        ]

        hallucination_ids = [None, 1, 2, None, None, None, 2]

        trie = ProbabilityModifierTrieNode.construct_root(tensor(probs), tensor(tok_ids), 3, hallucination_ids)

        raw_probs = np.array([0.3, .6 - .6, 0.1])
        np.testing.assert_almost_equal(trie.get_child([1, 1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.2, 0.5 - (.6 * .5), 0.3 - 0.3])
        np.testing.assert_almost_equal(trie.get_child([1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4 - (.6 * .5 * .4) - (.3 * .4), 0.1])
        np.testing.assert_almost_equal(trie.get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4, 0.1])
        np.testing.assert_almost_equal(trie.get_child([0]).get_probs().tolist(), raw_probs / raw_probs.sum())

    def test_parent_hallucination_child_first(self):
        tok_ids = [
            [0, 2, 0],
            [1, 2, 1],
            [1, 2, 0],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 0],
            [1, 1, 1],
        ]
        probs = [
            [[0.5, 0.4, 0.1]] * 4,
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]],
        ]

        hallucination_ids = [None, 2, 1, None, None, None, 2]

        trie = ProbabilityModifierTrieNode.construct_root(tensor(probs), tensor(tok_ids), 3, hallucination_ids)

        raw_probs = np.array([0.3, .6 - .6, 0.1])
        np.testing.assert_almost_equal(trie.get_child([1, 1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.2, 0.5 - (.6 * .5), 0.3 - 0.3])
        np.testing.assert_almost_equal(trie.get_child([1]).get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4 - (.6 * .5 * .4) - (.3 * .4), 0.1])
        np.testing.assert_almost_equal(trie.get_probs().tolist(), raw_probs / raw_probs.sum())

        raw_probs = np.array([0.5, 0.4, 0.1])
        np.testing.assert_almost_equal(trie.get_child([0]).get_probs().tolist(), raw_probs / raw_probs.sum())

    def test_probability_mass_trace(self):
        tok_ids = [
            [0, 2, 0],
            [1, 2, 0],
            [1, 0, 2],
            [1, 1, 2],
            [1, 1, 0],
            [1, 1, 1],
        ]
        probs = [
            [[0.5, 0.4, 0.1]] * 4,
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.5, 0.5, 0], [0.5, 0.5, 0]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.4, 0.1], [0.2, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]],
        ]
        trie = ProbabilityModifierTrieNode.construct_root(probs=tensor(probs),
                                                          sampled_toks=tensor(tok_ids),
                                                          eos_tok_id=2)
        np.testing.assert_almost_equal(trie.get_cumulative_prob_mass_trace(), [])
        np.testing.assert_almost_equal(trie.get_child([1, 1, 2]).get_cumulative_prob_mass_trace(), [0.4, 0.2, 0.02])
        np.testing.assert_almost_equal(trie.get_child([1, 1, 1]).get_cumulative_prob_mass_trace(), [0.4, 0.2, 0.12])
        np.testing.assert_almost_equal(trie.get_child([0, 2]).get_cumulative_prob_mass_trace(), [0.5, 0.05])

        trie.get_child([1, 1, 1]).mark_as_hallucination()
        # Removes 0.12 from probability trie
        a = trie.get_child([1, 1, 2]).get_cumulative_prob_mass_trace()
        b = [(0.4 - 0.12) / 0.88, (0.2 - 0.12) / 0.88, 0.02 / 0.88]
        np.testing.assert_almost_equal(trie.get_child([1, 1, 2]).get_cumulative_prob_mass_trace(),
                                       [(0.4 - 0.12) / 0.88, (0.2 - 0.12) / 0.88, 0.02 / 0.88])
        np.testing.assert_almost_equal(trie.get_child([1, 1, 1]).get_cumulative_prob_mass_trace(),
                                       [(0.4 - 0.12) / 0.88, (0.2 - 0.12) / 0.88, 0])
        np.testing.assert_almost_equal(trie.get_child([0, 2]).get_cumulative_prob_mass_trace(),
                                       [0.5 / 0.88, 0.05 / 0.88])

        trie.get_child([0, 2]).mark_as_hallucination()
        # Removes another 0.05 from probability trie
        np.testing.assert_almost_equal(trie.get_child([1, 1, 2]).get_cumulative_prob_mass_trace(),
                                       [(0.4 - 0.12) / 0.83, (0.2 - 0.12) / 0.83, 0.02 / 0.83])
        np.testing.assert_almost_equal(trie.get_child([1, 1, 1]).get_cumulative_prob_mass_trace(),
                                       [(0.4 - 0.12) / 0.83, (0.2 - 0.12) / 0.83, 0])
        np.testing.assert_almost_equal(trie.get_child([0, 2]).get_cumulative_prob_mass_trace(),
                                       [(0.5 - 0.05) / 0.83, 0])

    def test_resampling(self):
        tok_ids = [
            [0, 0, 0]
        ]

        probs = [
            [[1, 0, 0], [0.25, 0.25, 0.5], [1, 0, 0], [0.5, 0, 0.5]]
        ]

        trie = ProbabilityModifierTrieNode.construct_root(probs=tensor(probs), sampled_toks=tensor(tok_ids),
                                                          eos_tok_id=2)
        child = trie.get_child([0, 0, 0])
        prob_trace = child.get_selection_prob_trace()
        child.mark_child_as_hallucination(0)

        results = Counter()
        for i in range(10000):
            results[child.calc_probabilistic_backtracking(prob_trace)] += 1

        def assert_about_in_counter(key, val):
            res = results[key] / 10000
            self.assertGreaterEqual(res, 0.96 * val)
            self.assertLessEqual(res, 1.04 * val)

        prob_ratio = (.125 * 8 / 7) / .25
        # Prob mass of node decreased by half, then increased by 8/7 from renormalization
        assert_about_in_counter(None, prob_ratio)
        # Remainder of probability mass is distributed to other nodes
        assert_about_in_counter((1, 1), (1 - prob_ratio) / 3)
        assert_about_in_counter((1, 2), 2 * (1 - prob_ratio) / 3)
