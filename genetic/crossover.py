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

    def __call__(self, parents: list[tuple[Tensor, Tensor]], ctx: dict) -> list[Tensor]:
        """
        Apply the crossover operation to a list of parents.

        Args:
            parents: List of pairs of parents to crossover.
            ctx: Context dictionary for additional information.

        Returns:
            list: List of offspring generated from the crossover.
        """
        if self.method == "none":
            return [parent for pair in parents for parent in pair]

        offspring = []
        for pair in parents:
            if len(pair) != 2:
                raise ValueError("Crossover requires exactly two parents.")

            parent1, parent2 = pair
            if random.random() < 0.5:
                parent1, parent2 = parent2, parent1

            if parent1.device != parent2.device:
                raise ValueError("Parents must be on the same device.")

            if parent1.device is None:
                offspring.append(parent1)
            elif parent1.device == Device.CPU:
                offspring.append(apply_crossover_cpu(parent1, parent2, self.method))
            elif parent1.device == Device.GPU and apply_crossover_gpu is not None:
                offspring.append(
                    apply_crossover_gpu(parent1, parent2, self.method, ctx)
                )
            else:
                raise NotImplementedError(
                    f"Device {parent1.device} not supported for crossover."
                )

        return offspring
