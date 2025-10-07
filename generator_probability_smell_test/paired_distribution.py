import random
from typing import List, Set


class PairedDistribution:
    def __init__(self, main: List[float], speculative: List[float], hallus: Set[int] = None):
        self.main = main
        self.speculative = speculative
        assert len(main) == len(speculative)
        self.population = list(range(len(main)))
        self.hallus = hallus or set()

    def sample_main(self) -> int:
        return random.choices(self.population, weights=self.main)[0]

    def sample_speculative(self) -> int:
        return random.choices(self.population, weights=self.speculative)[0]
