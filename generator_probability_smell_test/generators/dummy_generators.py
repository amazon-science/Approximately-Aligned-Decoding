import random

from generator_probability_smell_test.paired_distribution import PairedDistribution
from generator_probability_smell_test.speculative_generator import SpeculativeGenerator


class PerfectGenerator(SpeculativeGenerator):
    """
    Directly samples from the main distribution
    """

    def generate(self, distribution: PairedDistribution, seed):
        return distribution.sample_main()


class IncorrectSpeculativeDistributionGenerator(SpeculativeGenerator):
    """
    Samples from the speculative distribution, rather than the main distribution
    """

    def generate(self, distribution: PairedDistribution, seed):
        return distribution.sample_speculative()


class RandomGenerator(SpeculativeGenerator):
    """
    Samples from a uniform random distribution
    """

    def generate(self, distribution: PairedDistribution, seed):
        return random.randint(0, len(distribution.main) - 1)
