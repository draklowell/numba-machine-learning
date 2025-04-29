import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import LinearUnit


class Linear(Layer):
    """
    A layer descriptor for the linear layer.
    This layer applies a linear transformation to the input tensor.
    """

    name: str = "linear"
    _output_size: int
    _include_bias: bool

    def __init__(self, output_size: int, include_bias: bool = True):
        super().__init__()

        self._output_size = output_size
        self._include_bias = include_bias

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> LinearUnit:
        return LinearUnit(
            name,
            shape,
            dtype,
            device,
            self._output_size,
            self._include_bias,
        )
