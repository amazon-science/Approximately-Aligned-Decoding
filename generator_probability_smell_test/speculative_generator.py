import abc
from typing import Any

from generator_probability_smell_test.paired_distribution import PairedDistribution


class SpeculativeGenerator(abc.ABC):

    @abc.abstractmethod
    def generate(self, distribution: PairedDistribution, seed: Any):
        pass
