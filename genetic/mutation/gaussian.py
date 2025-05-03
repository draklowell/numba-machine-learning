import numpy as np

from genetic.cpu.mutation_gaussian import apply_gaussian as apply_gaussian_cpu

try:
    from genetic.gpu.mutation_gaussian import apply_gaussian as apply_gaussian_gpu
except ImportError:
    apply_gaussian_gpu = None
from genetic.mutation.base import Mutation
from nml import Device, Parameter, Tensor


class GaussianMutation(Mutation):
    """
    Gaussian mutation for real-valued genomes.
    """

    parameter: Parameter
    rate: np.number
    strength: np.number

    def __init__(
        self,
        parameter: Parameter,
        rate: np.number = 0.01,
        strength: np.number = 0.05,
    ):
        self.parameter = parameter
        self.rate = rate
        self.strength = strength

    def __call__(self, offspring: Tensor, ctx: dict) -> Tensor:
        if offspring.device is None or offspring.device == Device.CPU:
            return apply_gaussian_cpu(
                offspring,
                self.parameter.low,
                self.parameter.high,
                self.rate,
                self.strength,
            )

        if offspring.device == Device.GPU and apply_gaussian_gpu is not None:
            return apply_gaussian_gpu(
                offspring,
                self.parameter.low,
                self.parameter.high,
                self.rate,
                self.strength,
                ctx=ctx,
            )

        raise NotImplementedError(
            f"Device {offspring.device} not supported for Gaussian mutation."
        )
