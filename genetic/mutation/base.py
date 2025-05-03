from abc import ABC, abstractmethod

from nml import Tensor


class Mutation(ABC):
    """
    Abstract base class for mutation operations.
    """

    @abstractmethod
    def __call__(self, offspring: list[Tensor], ctx: dict) -> list[Tensor]:
        """
        Apply the mutation operation to a list of offspring.

        Args:
            offspring: List of offspring to mutate.
            ctx: Context dictionary for additional information.

        Returns:
            list: List of mutated offspring.
        """
