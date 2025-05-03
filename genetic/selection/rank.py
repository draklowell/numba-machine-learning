import random
from typing import Any

from genetic.selection.base import Selection


class RankSelection(Selection):
    """
    Rank selection strategy for genetic algorithms.

    Randomly selects given number of individuals where rank compered to other genomes
    is a possibility of each individual to be selected.

    For example for candidates [(G1, 31), (G2, 28), (G3, 98), (G4, 42)] ranking will be:
    4-G3, 3-G4, 2-G1, 1-G2
    So the highest possibility to be selected will be in G3 (4/(4+3+2+1))
    """

    population_size: int

    def __init__(self, population_size: int):
        self.population_size = population_size

    def __call__(self, population: list[tuple[float, Any]]) -> list[tuple[float, Any]]:
        selected = []
        sorted_candidates = sorted(population, key=lambda x: x[1])

        ranks = []
        ranks_sum = 0
        for x in range(len(sorted_candidates), 0, -1):
            ranks.append(x)
            ranks_sum += x

        for _ in range(self.population_size):
            pick = random.uniform(0, ranks_sum)
            current = 0
            for candidate, rank_ in zip([c[0] for c in sorted_candidates], ranks):
                current += rank_
                if current >= pick:
                    selected.append(candidate)
                    break

        return selected
