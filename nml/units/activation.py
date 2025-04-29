import numpy as np

from nml.cpu.tensor import CPUTensor
from nml.device import Device
from nml.parameter import Parameter
from nml.tensor import Tensor
from nml.units.base import Unit, UnitWithWeights

try:
    from nml.gpu import apply_activation, apply_softmax
except ImportError:
    apply_activation = None
    apply_softmax = None


activations = {
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
}


class ActivationUnit(Unit):
    """
    Activation Unit for applying activation functions to a tensor.
    """

    _activation: str

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        activation: str,
    ):
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")

        if activation == "softmax" and len(shape) != 1:
            raise ValueError("Softmax activation requires a 1D tensor.")

        self._activation = activation
        super().__init__(name, shape, dtype, device)

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        match self.device:
            case Device.CPU:
                return CPUTensor(activations[self._activation](tensor.array))
            case Device.GPU if (
                self._activation == "softmax" and apply_softmax is not None
            ):
                return apply_softmax(tensor, ctx=ctx)
            case Device.GPU if apply_activation is not None:
                return apply_activation(self._activation, tensor, ctx=ctx)

        raise NotImplementedError(
            f"Device {self.device} is not supported for ActivationUnit."
        )


class LeakyReLUUnit(Unit):
    """
    Leaky ReLU Unit for applying the Leaky ReLU activation function to a tensor.
    """

    _alpha: np.number

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        alpha: np.number,
    ):
        self._alpha = alpha

        super().__init__(name, shape, dtype, device)

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        match self.device:
            case Device.CPU:
                return CPUTensor(
                    np.where(tensor.array > 0, tensor.array, self._alpha * tensor.array)
                )
            case Device.GPU if apply_activation is not None:
                return apply_activation("leaky_relu", tensor, self._alpha, ctx=ctx)

        raise NotImplementedError(
            f"Device {self.device} is not supported for LeakyReLUUnit."
        )


class PReLUUnit(UnitWithWeights):
    """
    PReLU Unit for applying the Parametric ReLU activation function to a tensor.
    """

    def __init__(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype, device: Device
    ):
        super().__init__(
            [
                Parameter(
                    name="alpha",
                    shape=(),
                    dtype=dtype,
                    low=0.0,
                    high=1.0,
                )
            ],
            name,
            shape,
            dtype,
            device,
        )

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        match self.device:
            case Device.CPU:
                return CPUTensor(
                    np.where(
                        tensor.array > 0,
                        tensor.array,
                        self._weights["alpha"].value * tensor.array,
                    )
                )
            case Device.GPU if apply_activation is not None:
                return apply_activation(
                    "leaky_relu", tensor, self._weights["alpha"].value, ctx=ctx
                )

        raise NotImplementedError(
            f"Device {self.device} is not supported for PReLUUnit."
        )
