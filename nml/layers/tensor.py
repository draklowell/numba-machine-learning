import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import CastUnit, FlattenUnit, ReshapeUnit


class Reshape(Layer):
    """
    A layer descriptor for the reshape layer.
    This layer reshapes the input tensor to a specified shape.
    """

    name: str = "reshape"
    _shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()

        self._shape = shape

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ReshapeUnit:
        return ReshapeUnit(
            name,
            shape,
            dtype,
            device,
            self._shape,
        )


class Flatten(Layer):
    """
    A layer descriptor for the flatten layer.
    This layer reshapes the input tensor to a 1D tensor.
    """

    name: str = "flatten"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> FlattenUnit:
        return FlattenUnit(
            name,
            shape,
            dtype,
            device,
        )


class Cast(Layer):
    """
    A layer descriptor for the cast layer.
    This layer casts the input tensor to a specified dtype.
    """

    name: str = "cast"
    _dtype: np.dtype

    def __init__(self, dtype: np.dtype):
        super().__init__()

        self._dtype = dtype

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> CastUnit:
        return CastUnit(
            name,
            shape,
            dtype,
            device,
            self._dtype,
        )
