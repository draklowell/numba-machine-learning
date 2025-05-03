from abc import ABC, abstractmethod
from typing import Any


class Selection(ABC):
    """
    Abstract base class for selection strategies in genetic algorithms.
    """

    @abstractmethod
    def __call__(self, population: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        """
        Select individuals from the population based on the selection strategy.

        Args:
            population: The population to select from, where each individual is represented as a
                        tuple containing its individual itself and the fitness.

        Returns:
            The selected individuals, represented as a list of tuples containing the individual
            and their fitness.
        """
