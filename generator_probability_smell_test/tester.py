from typing import Set, Optional, Tuple, Sequence

import numpy as np
import torch
from ansi.colour import fg, bg
from scipy.stats import chisquare

from approx_aligned_decoding.backtracking_strategy.fast_gad_backtracking import FastGADBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.grammar_aligned_decoding_backtracking import \
    GrammarAlignedDecodingBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.naive_backtracking import NaiveBacktrackingStrategy
from generator_probability_smell_test.generators.custom_generator import CustomGenerator, CustomGeneratorSequence, \
    CustomGeneratorNonSpeculative, CustomGeneratorSequenceNonSpeculative
from generator_probability_smell_test.generators.dummy_generators import PerfectGenerator, RandomGenerator, \
    IncorrectSpeculativeDistributionGenerator
from generator_probability_smell_test.generators.huggingface_transformers_generator import \
    TransformersGeneratorNonSpeculative, TransformersGenerator, TransformersGeneratorSequenceNonSpeculative, \
    TransformersGeneratorSequenceSpeculative
from generator_probability_smell_test.paired_distribution import PairedDistribution
from generator_probability_smell_test.speculative_generator import SpeculativeGenerator


def test_generates_correct_distribution(generator: SpeculativeGenerator, distribution: PairedDistribution,
                                        n_samples: int) -> Tuple[float, Sequence[float], Sequence[int]]:
    # Null hypothesis: speculative generator has same distribution as main generator.
    # Alternative hypothesis: speculative generator has different distribution from main generator.
    # p-value: probability of seeing a result as extreme as the one observed.

    # This is a bit silly, but otherwise we get annoying float errors (grrrr...)
    expected_dist = np.asarray(distribution.main) + 1e-50

    if distribution.hallus is not None:
        for hallu_idx in distribution.hallus:
            expected_dist[hallu_idx] = 1e-50

    expected_dist = (expected_dist * n_samples) / expected_dist.sum()

    actual_dist = [0 for _ in distribution.main]
    for i in range(n_samples):
        generated_val = generator.generate(distribution, i)
        actual_dist[generated_val] += 1

    return chisquare(f_obs=actual_dist, f_exp=expected_dist).pvalue, expected_dist, actual_dist


tests = [
    ("identical_dists", [.1] * 10, [.1] * 10),
    ("disjoint_dists", [.2] * 5 + [0] * 5, [0] * 5 + [.2] * 5),
    ("example_1", [0.1, 0.9], [0.1, 0.9]),
    ("example_2", [0.1, 0.8, 0.1], [0.4, 0.6, 0]),
    ("fuzz_1", [0.004720211029052734, 0.9302527904510498, 0.7257322072982788, 0.8294872641563416, 0.7682948708534241],
     [0.06001889705657959, 0.14530813694000244, 0.29239821434020996, 0.529152512550354, 0.14655780792236328]),
    ("fuzz_2", [0.6569646596908569, 0.002136528491973877, 0.4128659963607788, 0.23163312673568726, 0.42301130294799805],
     [0.560369610786438, 0.8256030082702637, 0.884706974029541, 0.39371973276138306, 0.5363077521324158])
]

hallu_tests = [
    ("uniform most hallu", [1 / 27] * 27, [1 / 27] * 27, set(range(27)).difference({0, 1, 9})),
    ("uniform one hallu", [1 / 27] * 27, [1 / 27] * 27, {0}),
]

def run_test(generator: SpeculativeGenerator, name: str, main_dist: list[float], spec_dist: list[float],
             n_samples: int, hallu_indices: Optional[Set[int]] = None):
    distribution = PairedDistribution(main_dist, spec_dist, hallu_indices)

    pvalue, expected, actual = test_generates_correct_distribution(generator, distribution, n_samples)

    if pvalue < 0.01:
        color = lambda x: x
        if pvalue < 0.001:
            color = fg.yellow
        if pvalue < 0.0001:
            color = fg.red
        if pvalue < FAIL_P_VALUE:
            color = lambda x: fg.black(bg.red(x))
            print(f"expected/actual: {list(enumerate(zip(expected, actual)))}")
        print(f"{name}: p={color('{:e}'.format(pvalue))}")

    return pvalue, expected, actual


FAIL_P_VALUE = 1e-7


def run_all_tests(generator: SpeculativeGenerator, n_samples: int):
    for name, main_dist, spec_dist in tests:
        pvalue, expected, actual = run_test(generator, name, main_dist, spec_dist, n_samples)
        if pvalue < FAIL_P_VALUE:
            print(f"Failed test {name}\n")
            return False

    for i in range(20):
        pvalue = random_test(generator, i, 10, n_samples, True, False)
        if pvalue < FAIL_P_VALUE:
            print(f"Failed random test {i}\n")
            return False

    print(f"Passed all tests!\n")


backtrack_strategies = [
    ("Naive (expected to fail!)", NaiveBacktrackingStrategy()),
    ("GAD", GrammarAlignedDecodingBacktrackingStrategy()),
    ("FastGAD (expected to fail!)", FastGADBacktrackingStrategy()),
]


def run_seq_tests(generator, n_samples: int, hallu_detect: bool):
    for i in range(20):
        if hallu_detect:
            gen = generator(None)
        else:
            gen = generator

        pvalue = random_test(gen, i, 27, n_samples, False, False)
        if pvalue < FAIL_P_VALUE:
            print(f"Failed random test {i}\n")
            break
    else:
        print("Passed random tests!\n")

    if hallu_detect:
        for strat_name, strategy in backtrack_strategies:
            print(f"{strat_name}:")
            for name, main_dist, spec_dist, hallu_indices in hallu_tests:
                pvalue, expected, actual = run_test(generator(strategy), name, main_dist, spec_dist, n_samples,
                                                    hallu_indices)
                if pvalue < FAIL_P_VALUE:
                    print(f"Failed test {name}\n")
                    break

            for i in range(20):
                pvalue = random_test(generator(strategy), i, 27, n_samples, False, True)
                if pvalue < FAIL_P_VALUE:
                    print(f"Failed random test {i}\n")
                    break
            else:
                print(f"Passed {strat_name} tests!\n")


generators = [
    ("Perfect Generator", PerfectGenerator()),
    ("Sample from wrong dist (Expected to fail!)", IncorrectSpeculativeDistributionGenerator()),
    ("Random Generator (Expected to fail!)", RandomGenerator()),
    ("Transformers Generator (non-speculative)", TransformersGeneratorNonSpeculative()),
    ("Transformers Generator (speculative)", TransformersGenerator()),
    ("Our Generator (speculative)", CustomGenerator()),
    ("Our Generator (non-speculative)", CustomGeneratorNonSpeculative()),
]

seq_generators = [
    ("Our Generator (non-speculative)", lambda b: CustomGeneratorSequenceNonSpeculative(3, 3, b), True),
    ("Our Generator (speculative)", lambda b: CustomGeneratorSequence(3, 3, b), True),
    ("Transformers Sequence Generator (non-speculative)", TransformersGeneratorSequenceNonSpeculative(3, 3), False),
    ("Transformers Sequence Generator (speculative)", TransformersGeneratorSequenceSpeculative(3, 3), False),
]


def random_test(generator: SpeculativeGenerator, seed: int, max_num_tokens: int, n_samples: int, var_num_toks: bool,
                hallu: bool):
    torch.manual_seed(seed)

    num_tokens = torch.randint(max_num_tokens, (1,)).item() + 1 if var_num_toks else max_num_tokens
    main_distr = (torch.rand(num_tokens) * 3).softmax(dim=0).numpy().tolist()
    spec_distr = (torch.rand(num_tokens) * 3).softmax(dim=0).numpy().tolist()

    if hallu:
        num_hallus = torch.randint(1, num_tokens - 1, (1,)).item()
        # Some might get discarded when creating the set due to duplicates but good enough
        hallu_indices = set(torch.randint(0, num_tokens, (num_hallus,)).tolist())
    else:
        hallu_indices = None

    pvalue, expected, actual = run_test(generator, f"Random test {seed}", main_distr, spec_distr, n_samples,
                                        hallu_indices)
    if pvalue < FAIL_P_VALUE:
        print(main_distr)
        print(spec_distr)
        if hallu:
            print(hallu_indices)

    return pvalue


def run_tests():
    for name, generator, hallu_detect in seq_generators:
        print(f"{name}:")
        run_seq_tests(generator, 1000, hallu_detect)
        print()

    for name, generator in generators:
        print(f"{name}:")
        run_all_tests(generator, 1000)
        print()


if __name__ == '__main__':
    run_tests()
