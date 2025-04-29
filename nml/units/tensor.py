import numpy as np

from nml.device import Device
from nml.tensor import Tensor
from nml.units.base import Unit


class ReshapeUnit(Unit):
    """
    Reshape Unit for reshaping a tensor to a specified shape.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        output_shape: tuple[int, ...],
    ):
        if np.prod(shape) != np.prod(output_shape):
            raise ValueError(
                f"Cannot reshape tensor of shape {shape} to {output_shape}"
            )

        super().__init__(name, output_shape, dtype, device)
        self._output_shape = output_shape

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        return tensor.reshape((tensor.shape[0],) + self._output_shape, ctx=ctx)


class FlattenUnit(ReshapeUnit):
    """
    Flatten Unit for reshaping a tensor to a specified shape.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
    ):
        super().__init__(name, shape, dtype, device, (int(np.prod(shape)),))


class CastUnit(Unit):
    """
    Cast Unit for casting a tensor to a specified dtype.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        output_dtype: np.dtype,
    ):
        if not np.can_cast(dtype, output_dtype, casting="safe"):
            raise ValueError(f"Cannot cast {dtype} to {output_dtype}")

        super().__init__(name, shape, output_dtype, device)
        self._output_dtype = output_dtype

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        return tensor.cast(self._output_dtype, ctx=ctx)
