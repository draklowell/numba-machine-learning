import numpy as np
from numba import cuda

from nml.device import Device
from nml.tensor import Tensor


class GPUTensor(Tensor):
    """
    A tensor that resides on the GPU.
    This class is used to represent tensors that are stored in GPU memory.
    """

    device: Device = Device.GPU

    def __init__(self, array):
        self.array = array

    @classmethod
    def empty(
        cls, shape: tuple[int, ...] | int, dtype: np.dtype, ctx: dict | None = None
    ) -> "GPUTensor":
        if ctx is not None:
            stream = ctx.get("cuda.stream")
        else:
            stream = None

        return GPUTensor(cuda.device_array(shape, dtype=dtype, stream=stream))

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
    ) -> "GPUTensor":
        return GPUTensor(self.array.reshape(shape))

    def cast(self, dtype: np.dtype, ctx: dict | None = None) -> "GPUTensor":
        from nml.gpu.cast import apply_cast

        return apply_cast(self, dtype, ctx)
