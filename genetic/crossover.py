import random
from typing import Literal

from genetic.cpu.crossover import apply_crossover as apply_crossover_cpu
from nml import Device, Tensor

try:
    from genetic.gpu.crossover import apply_crossover as apply_crossover_gpu
except ImportError:
    apply_crossover_gpu = None


class Crossover:
    """
    Crossover operation for genetic algorithms.
    """

    method: Literal["single_point", "two_point", "uniform", "none"]

    def __init__(
        self,
        method: Literal[
            "single_point", "two_point", "uniform", "none"
        ] = "single_point",
    ):
        self.method = method

    def __call__(self, parents: tuple[Tensor, Tensor], ctx: dict) -> Tensor:
        """
        Apply the crossover operation to a list of parents.

        Args:
            parents: Pair of parents to crossover.
            ctx: Context dictionary for additional information.

        Returns:
            Offspring generated from the crossover.
        """
        if len(parents) != 2:
            raise ValueError("Crossover requires exactly two parents.")
        parent1, parent2 = parents

        if random.random() < 0.5:
            parent1, parent2 = parent2, parent1

        if parent1.device != parent2.device:
            raise ValueError("Parents must be on the same device.")

        if self.method == "none":
            return parent1

        if parent1.device is None:
            return parent1

        if parent1.device == Device.CPU:
            return apply_crossover_cpu(parent1, parent2, self.method)

        if parent1.device == Device.GPU and apply_crossover_gpu is not None:
            return apply_crossover_gpu(parent1, parent2, self.method, ctx)

        raise NotImplementedError(
            f"Device {parent1.device} not supported for crossover."
        )
