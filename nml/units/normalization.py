import numpy as np

from nml.device import Device
from nml.tensor import Tensor
from nml.units.base import Unit

try:
    from nml.gpu import apply_normalization
except ImportError:
    apply_normalization = None


class NormalizeUnit(Unit):
    """
    Normalize Unit for applying normalization to a tensor.
    This unit applies a linear transformation to the input tensor
    using the formula: y = k * x + b, where k and b are constants.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        k: float,
        b: float,
    ):
        self.k = k
        self.b = b
        super().__init__(name, shape, dtype, device)

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        match self.device:
            case Device.CPU:
                tensor.array *= self.k
                tensor.array += self.b
                return tensor
            case Device.GPU if apply_normalization is not None:
                return apply_normalization(
                    tensor,
                    self.k,
                    self.b,
                    ctx=ctx,
                )

        raise NotImplementedError(
            f"Device {self.device} is not supported for NormalizeUnit."
        )
