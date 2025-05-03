import random
from typing import Any

from genetic.selection.base import Selection


class TournamentSelection(Selection):
    """
    Tournament selection strategy for genetic algorithms.

    Randomly selects tournament_size number of candidates and chooses the best among them.
    Repeats so until it gets a resulting list with the given size.
    """

    population_size: int
    tournament_size: int

    def __init__(self, population_size: int, tournament_size: int = 2):
        self.population_size = population_size
        self.tournament_size = tournament_size

    def __call__(self, population: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner)

        return selected
