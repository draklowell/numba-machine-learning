import numpy as np

from genetic.cpu.mutation_gaussian import apply_gaussian as apply_gaussian_cpu
from genetic.mutation.base import Mutation
from nml import Device, Parameter, Tensor


class GaussianScaledMutation(Mutation):
    """
    Gaussian scaled mutation for genomes, where mutation strength is scaled by the
    standard deviation of the genome.
    """

    parameter: Parameter
    scale: np.number
    rate: np.number
    min_strength: np.number

    def __init__(
        self,
        parameter: Parameter,
        scale: np.number = 0.01,
        rate: np.number = 0.01,
        min_strength: np.number = 0.05,
    ):
        self.parameter = parameter
        self.scale = scale
        self.rate = rate
        self.min_strength = min_strength

    def __call__(self, offspring: Tensor, ctx: dict) -> Tensor:
        if offspring.device == Device.CPU:
            strength = self.scale * np.clip(
                np.std(offspring.array), self.min_strength, None
            )

            return apply_gaussian_cpu(
                offspring,
                self.parameter.low,
                self.parameter.high,
                self.rate,
                strength,
            )

        raise NotImplementedError(
            f"Device {offspring.device} not supported for Gaussian scaled mutation."
        )
