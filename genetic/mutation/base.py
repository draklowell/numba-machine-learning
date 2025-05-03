from abc import ABC, abstractmethod

from nml import Tensor


class Mutation(ABC):
    """
    Abstract base class for mutation operations.
    """

    @abstractmethod
    def __call__(self, offspring: Tensor, ctx: dict) -> Tensor:
        """
        Apply the mutation operation to a list of offspring.

        Args:
            offspring: Tensor representing the offspring to be mutated.
            ctx: Context dictionary for additional information.

        Returns:
            The mutated offspring tensor.
        """
