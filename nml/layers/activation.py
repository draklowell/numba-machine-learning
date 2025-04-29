import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import ActivationUnit


class Softmax(Layer):
    """
    A layer descriptor for the softmax layer.
    This layer applies the softmax activation function to the input tensor.
    """

    name: str = "softmax"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ActivationUnit:
        return ActivationUnit(
            name,
            shape,
            dtype,
            device,
            "softmax",
        )


class Sigmoid(Layer):
    """
    A layer descriptor for the sigmoid layer.
    This layer applies the sigmoid activation function to the input tensor.
    """

    name: str = "sigmoid"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ActivationUnit:
        return ActivationUnit(
            name,
            shape,
            dtype,
            device,
            "sigmoid",
        )


class Tanh(Layer):
    """
    A layer descriptor for the tanh layer.
    This layer applies the tanh activation function to the input tensor.
    """

    name: str = "tanh"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ActivationUnit:
        return ActivationUnit(
            name,
            shape,
            dtype,
            device,
            "tanh",
        )


class ReLU(Layer):
    """
    A layer descriptor for the ReLU layer.
    This layer applies the ReLU activation function to the input tensor.
    """

    name: str = "relu"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ActivationUnit:
        return ActivationUnit(
            name,
            shape,
            dtype,
            device,
            "relu",
        )
