import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import ActivationUnit, LeakyReLUUnit, PReLUUnit


class Softmax(Layer):
    """
    A layer descriptor for the softmax layer.
    This layer applies the softmax activation function to the input tensor.
    """

    name: str = "softmax"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> ActivationUnit:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

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
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

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
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

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


class PReLU(Layer):
    """
    A layer descriptor for the PReLU layer.
    This layer applies the PReLU activation function to the input tensor.
    """

    name: str = "prelu"

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> PReLUUnit:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

        return PReLUUnit(
            name,
            shape,
            dtype,
            device,
        )


class LeakyReLU(Layer):
    """
    A layer descriptor for the Leaky ReLU layer.
    This layer applies the Leaky ReLU activation function to the input tensor.
    """

    name: str = "leaky_relu"
    _alpha: np.number

    def __init__(self, alpha: np.number = 0.01):
        self._alpha = alpha

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> LeakyReLUUnit:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

        return LeakyReLUUnit(
            name,
            shape,
            dtype,
            device,
            self._alpha,
        )
