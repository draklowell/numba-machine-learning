from typing import Any

from genetic.selection.base import Selection


class BestSelection(Selection):
    """
    Selects the best individuals from the population based on their fitness scores.
    This selection strategy retains the top N individuals with the highest fitness scores.
    """

    population_size: int

    def __init__(self, population_size: int):
        self.population_size = population_size

    def __call__(self, population: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        sorted_candidates = sorted(population, key=lambda x: x[1])
        return sorted_candidates[: self.population_size]
