import numpy as np

from nml.device import Device
from nml.tensor import Tensor


class CPUTensor(Tensor):
    """
    A tensor that resides on the CPU.
    This class is used to represent tensors that are stored in CPU memory.
    """

    device: Device = Device.CPU

    def __init__(self, array: np.ndarray):
        self.array = array

    @classmethod
    def empty(
        cls, shape: tuple[int, ...] | int, dtype: np.dtype, ctx: dict | None = None
    ) -> "CPUTensor":
        return CPUTensor(np.empty(shape, dtype=dtype))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def reshape(
        self, shape: tuple[int, ...] | int, ctx: dict | None = None
    ) -> "CPUTensor":
        return CPUTensor(self.array.reshape(shape))

    def cast(self, dtype: np.dtype, ctx: dict | None = None) -> "CPUTensor":
        return CPUTensor(self.array.astype(dtype))
