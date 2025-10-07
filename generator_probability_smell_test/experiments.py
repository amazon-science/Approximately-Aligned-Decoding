# Larger scale experiments to show outputs
import numpy as np
import scipy.stats

from approx_aligned_decoding.backtracking_strategy.fast_gad_backtracking import FastGADBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.grammar_aligned_decoding_backtracking import \
    GrammarAlignedDecodingBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.naive_backtracking import NaiveBacktrackingStrategy
from generator_probability_smell_test.generators.custom_generator import CustomGeneratorSequenceNonSpeculative
from generator_probability_smell_test.tester import run_test


def run_experiments():
    n_samples = 10000

    generator = lambda b: CustomGeneratorSequenceNonSpeculative(3, 3, b)
    strategies = [
        ("ASAp", GrammarAlignedDecodingBacktrackingStrategy()),
        ("Constrained", NaiveBacktrackingStrategy()),
        ("Approx", FastGADBacktrackingStrategy()),
    ]

    experiments = [
        ("$\emptyset$", [1 / 27] * 27, {}),
        ("AAA", [1 / 27] * 27, {0}),
        ("AAA, AAC", [1 / 27] * 27, {0, 2}),
        ("AAA, ACC", [1 / 27] * 27, {0, 8}),
        ("AAA, CCC", [1 / 27] * 27, {0, 26}),
        ("AAA, AAB, ABA, BAA", [1 / 27] * 27, {0, 1, 3, 9}),
        ("AAA, AAB, ABA, ABB, ABC, ACA, ACB, ACC", [1 / 27] * 27, {0, 1, 3, 4, 5, 6, 7, 8}),
        ("All except AAA, AAB, ABA, BAA", [1 / 27] * 27, set(range(27)).difference({0, 1, 3, 9})),
        ("All except AAA, BAA", [1 / 27] * 27, set(range(27)).difference({0, 9})),
    ]

    results = []

    for experiment_name, experiment_dist, hallu_indices in experiments:
        exp_result = []
        for strategy_name, strategy in strategies:
            g = generator(strategy)
            pvalue, expected, actual = run_test(g, experiment_name + " " + strategy_name, experiment_dist,
                                                experiment_dist, n_samples,
                                                hallu_indices)
            expected = np.asarray(expected)
            actual = np.asarray(actual)
            expected = expected / expected.sum()
            actual = actual / actual.sum()
            ent = float(scipy.stats.entropy(actual, expected))
            ratio = g.total_toks_model / g.total_output_toks
            exp_result.append((ent, ratio))
            print(experiment_name + " " + strategy_name + " " + str(ent) + " " + str(ratio))

        results.append(exp_result)

    for exp_result, (experiment_name, _, _) in zip(results, experiments):
        exp_result_strs = list(map(lambda x: f"{x[0]:.4f} & {x[1]:.3f}", exp_result))
        print(experiment_name + " & " + " & ".join(exp_result_strs) + "\\\\")


if __name__ == '__main__':
    run_experiments()
