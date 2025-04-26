import numpy as np
from numpy.typing import NDArray

from nml.core.gpu.linear import apply_linear_gpu
from nml.layers.base import InferableLayer, Layer
from nml.parameters import TensorParameter


class InferableLinear(InferableLayer):
    """
    A class representing an inferable linear layer.
    """

    def __init__(
        self,
        name: str,
        input_size: int,
        output_size: int,
        include_bias: bool,
        dtype: np.dtype,
    ):
        parameters = [
            TensorParameter(
                name="weights",
                shape=(input_size, output_size),
                dtype=dtype,
            ),
        ]
        if include_bias:
            parameters.append(
                TensorParameter(
                    name="biases",
                    shape=(output_size,),
                    dtype=dtype,
                )
            )
        super().__init__(name, parameters)
        self._output_size = output_size
        self._include_bias = include_bias
        self._dtype = dtype

    def infer(self, x: NDArray) -> NDArray:
        x = x @ self._get_parameter("weights")
        if self._include_bias:
            x += self._get_parameter("biases")
        return x

    def infer_cuda(self, x, stream):
        if self._include_bias:
            biases = self._get_parameter("biases")
        else:
            biases = np.zeros(self._output_size, dtype=self._dtype)

        return apply_linear_gpu(x, self._get_parameter("weights"), biases, stream)


class Linear(Layer):
    """
    Linear layer for applying a linear transformation to the input tensor.
    """

    name = "linear"

    def __init__(
        self,
        size: int,
        include_bias: bool = True,
        dtype: np.dtype = np.float32,
    ):
        super().__init__()
        self._size = size
        self._include_bias = include_bias
        self._dtype = dtype

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableLinear, tuple[int, ...], np.dtype]:
        if len(shape) != 1:
            raise ValueError(f"Linear layer only supports 1D input shape, got {shape}")

        return (
            InferableLinear(
                f"{self.name}_{idx}",
                np.prod(shape),
                self._size,
                self._include_bias,
                self._dtype,
            ),
            (self._size,),
            self._dtype,
        )
