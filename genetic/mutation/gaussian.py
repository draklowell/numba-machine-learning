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

    def __call__(self, offspring: list[Tensor], ctx: dict) -> list[Tensor]:
        for idx, tensor in enumerate(offspring):
            if tensor.device is None or tensor.device == Device.CPU:
                offspring[idx] = apply_gaussian_cpu(
                    tensor,
                    self.parameter.low,
                    self.parameter.high,
                    self.rate,
                    self.strength,
                )
            elif tensor.device == Device.GPU and apply_gaussian_gpu is not None:
                offspring[idx] = apply_gaussian_gpu(
                    tensor,
                    self.parameter.low,
                    self.parameter.high,
                    self.rate,
                    self.strength,
                    ctx=ctx,
                )
            else:
                raise NotImplementedError(
                    f"Device {tensor.device} not supported for Gaussian mutation."
                )

        return offspring
