import random
from typing import Any

from genetic.selection.base import Selection


class RouletteSelection(Selection):
    """
    Roulette selection strategy for genetic algorithms.

    Randomly selects given number of individuals where fitness is
    a possibility of each individual to be selected.
    """

    population_size: int

    def __init__(self, population_size: int):
        self.population_size = population_size

    def __call__(self, population: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        selected = []
        total_fitness = 0

        for _, fitness in population:
            total_fitness += fitness

        for _ in range(self.population_size):
            pick = random.uniform(0, total_fitness)
            current = 0
            for candidate, fitness in population:
                current += fitness
                if current >= pick:
                    selected.append((candidate, fitness))
                    break

        return selected
