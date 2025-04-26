from typing import Callable

import numpy as np
from numpy.typing import NDArray

from nml.core.gpu.activation import (
    apply_activation_gpu,
    apply_softmax_gpu,
    leaky_relu_kernel,
    relu_kernel,
    sigmoid_kernel,
    tanh_kernel,
)
from nml.layers.base import InferableLayer, Layer
from nml.parameters import TensorParameter


class InferableCallableLayer(InferableLayer):
    """
    A class representing an inferable layer that calls external function.
    """

    def __init__(
        self, name: str, func: Callable[[NDArray], NDArray], func_cuda: Callable = None
    ):
        super().__init__(name)
        self._func = func
        self._func_cuda = func_cuda

    def infer(self, x: NDArray) -> NDArray:
        return self._func(x)

    def infer_cuda(self, x, stream):
        if self._func_cuda is not None:
            return self._func_cuda(x, stream)

        super().infer_cuda(x, stream)


class Softmax(Layer):
    """
    Softmax activation layer.
    """

    name = "softmax"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCallableLayer, tuple[int, ...], np.dtype]:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "Tanh activation layer requires floating point data type. "
                "Please use a compatible data type."
            )

        return (
            InferableCallableLayer(f"{self.name}_{idx}", self._infer, self._infer_cuda),
            shape,
            dtype,
        )

    def _infer(self, x: NDArray) -> NDArray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _infer_cuda(self, x: NDArray, stream):
        return apply_softmax_gpu(x, stream)


class Tanh(Layer):
    """
    Tanh activation layer.
    """

    name = "tanh"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCallableLayer, tuple[int, ...], np.dtype]:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "Tanh activation layer requires floating point data type. "
                "Please use a compatible data type."
            )

        return (
            InferableCallableLayer(f"{self.name}_{idx}", np.tanh, self._infer_cuda),
            shape,
            dtype,
        )

    def _infer_cuda(self, x, stream):
        return apply_activation_gpu(tanh_kernel, x, stream=stream)


class LeakyReLU(Layer):
    """
    Leaky ReLU activation layer.
    """

    name = "leaky_relu"

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCallableLayer, tuple[int, ...], np.dtype]:
        if isinstance(self.alpha, float) and not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Alpha must be a float for dtype {dtype}. "
                f"Please use a compatible data type."
            )

        return (
            InferableCallableLayer(f"{self.name}_{idx}", self._infer, self._infer_cuda),
            shape,
            dtype,
        )

    def _infer(self, x: NDArray) -> NDArray:
        return np.where(x > 0, x, self.alpha * x)

    def _infer_cuda(self, x, stream):
        return apply_activation_gpu(leaky_relu_kernel, x, self.alpha, stream=stream)


class InferablePReLU(InferableLayer):
    """
    A class representing an inferable PReLU layer.
    """

    def __init__(self, name: str):
        super().__init__(
            name,
            [
                TensorParameter(
                    name="alpha",
                    shape=(),
                    dtype=np.float32,
                    low=0.0,
                    high=1.0,
                )
            ],
        )

    def infer(self, x: NDArray) -> NDArray:
        return np.where(x > 0, x, self._get_parameter("alpha") * x)

    def infer_cuda(self, x, stream):
        return apply_activation_gpu(
            leaky_relu_kernel,
            x,
            self._get_parameter("alpha").item(),
            stream=stream,
        )


class PReLU(Layer):
    """
    Parametric ReLU activation layer.
    """

    name = "prelu"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferablePReLU, tuple[int, ...], np.dtype]:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "PReLU activation layer requires floating point data type. "
                "Please use a compatible data type."
            )

        return InferablePReLU(f"{self.name}_{idx}"), shape, dtype


class ReLU(Layer):
    """
    ReLU activation layer.
    """

    name = "relu"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCallableLayer, tuple[int, ...], np.dtype]:
        return (
            InferableCallableLayer(f"{self.name}_{idx}", self._infer, self._infer_cuda),
            shape,
            dtype,
        )

    def _infer(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

    def _infer_cuda(self, x, stream):
        return apply_activation_gpu(relu_kernel, x, stream=stream)


class Sigmoid(Layer):
    """
    Sigmoid activation layer.
    """

    name = "sigmoid"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCallableLayer, tuple[int, ...], np.dtype]:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "Sigmoid activation layer requires floating point data type. "
                "Please use a compatible data type."
            )

        return (
            InferableCallableLayer(f"{self.name}_{idx}", self._infer, self._infer_cuda),
            shape,
            dtype,
        )

    def _infer(self, x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-x))

    def _infer_cuda(self, x, stream):
        return apply_activation_gpu(sigmoid_kernel, x, stream=stream)
