import numpy as np

from nml.layers.base import InferableLayer, Layer


class InferableFlatten(InferableLayer):
    """
    A class representing an inferable flatten layer.
    """

    def infer(self, x: np.ndarray) -> np.ndarray:
        if x.shape == ():
            return x
        return x.reshape(x.shape[0], -1)


class Flatten(Layer):
    """
    Flatten layer for reshaping input tensors.
    """

    name = "flatten"

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableFlatten, tuple[int, ...], np.dtype]:
        if shape == ():
            return shape, dtype

        return InferableFlatten(f"{self.name}_{idx}"), (np.prod(shape),), dtype


class InferableReshape(InferableLayer):
    """
    A class representing an inferable reshape layer.
    """

    _shape: tuple[int, ...]

    def __init__(self, name: str, shape: tuple[int, ...]):
        super().__init__(name)
        self._shape = shape

    def infer(self, x: np.ndarray) -> np.ndarray:
        if x.shape == ():
            return x
        return x.reshape(x.shape[:1] + self._shape)


class Reshape(Layer):
    """
    Reshape layer for reshaping input tensors.
    """

    name = "reshape"

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self._shape = shape

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableReshape, tuple[int, ...], np.dtype]:
        if np.prod(shape) != np.prod(self._shape):
            raise ValueError(
                f"Cannot reshape tensor of shape {shape} to {self._shape}. "
                f"Total number of elements must match."
            )

        return InferableReshape(f"{self.name}_{idx}", self._shape), self._shape, dtype


class InferableCast(InferableLayer):
    """
    A class representing an inferable cast layer.
    """

    _dtype: np.dtype

    def __init__(self, name: str, dtype: np.dtype):
        super().__init__(name)
        self._dtype = dtype

    def infer(self, x: np.ndarray) -> np.ndarray:
        return x.astype(self._dtype)


class Cast(Layer):
    """
    Cast layer for changing the data type of input tensors.
    """

    name = "cast"

    def __init__(self, dtype: np.dtype):
        super().__init__()
        self._dtype = dtype

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[tuple[int, ...], np.dtype]:
        if not np.can_cast(dtype, self._dtype):
            raise TypeError(
                f"Cannot cast {dtype} to {self._dtype}. "
                f"Please use a compatible data type."
            )

        return InferableCast(f"{self.name}_{idx}", self._dtype), shape, self._dtype
