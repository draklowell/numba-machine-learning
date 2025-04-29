from abc import ABC, abstractmethod

import numpy as np

from nml.device import Device


class Tensor(ABC):
    """
    Base class for tensor objects in the NML framework.
    This class provides an interface for tensor properties.
    """

    device: Device

    @classmethod
    @abstractmethod
    def empty(
        cls, shape: tuple[int, ...] | int, dtype: np.dtype, ctx: dict | None = None
    ) -> "Tensor":
        """
        Creates an empty tensor with the specified shape and data type.

        Args:
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            ctx: Optional context for the operation.
        Returns:
            An empty tensor with the specified shape and data type.
        """

    @classmethod
    def empty_like(cls, tensor: "Tensor", ctx: dict | None = None) -> "Tensor":
        """
        Creates an empty tensor with the same shape and data type as the given tensor.

        Args:
            tensor: The tensor to copy the shape and data type from.
            ctx: Optional context for the operation.
        Returns:
            An empty tensor with the same shape and data type as the given tensor.
        """
        return cls.empty(tensor.shape, tensor.dtype, ctx=ctx)

    @classmethod
    @abstractmethod
    def create(cls, list_: list, dtype: np.dtype, ctx: dict | None = None) -> "Tensor":
        """
        Creates a tensor from a list.

        Args:
            list_: The list to create the tensor from.
            dtype: The data type of the tensor.
            ctx: Optional context for the operation.
        Returns:
            A tensor created from the list.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the tensor.
        """

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the data type of the tensor.
        """

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the tensor.
        """
        return len(self.shape)

    @abstractmethod
    def reshape(
        self, shape: tuple[int, ...] | int, ctx: dict | None = None
    ) -> "Tensor":
        """
        Reshapes the tensor to the specified shape.

        Warning:
            This operation usually does not create a new tensor, it
            only changes the view of the existing tensor.

        Args:
            shape: The new shape of the tensor.
            ctx: Optional context for the operation.
        Returns:
            Reshaped tensor.
        """

    @abstractmethod
    def cast(self, dtype: np.dtype, ctx: dict | None = None) -> "Tensor":
        """
        Casts the tensor to a different data type.
        Args:
            dtype: The target data type.
            ctx: Optional context for the operation.
        Returns:
            A new tensor with the specified data type.
        """

    def __repr__(self):
        return f"{type(self).__name__}[{self.device}](shape={self.shape!r}, dtype={self.dtype!r})"


class Scalar(Tensor):
    """
    A scalar tensor.
    This class represents a scalar tensor with a single value.
    """

    device: Device = None

    def __init__(self, value: np.number, dtype: np.dtype):
        self.value = dtype.type(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @classmethod
    def empty(
        cls, shape: tuple[int, ...] | int, dtype: np.dtype, ctx: dict | None = None
    ) -> "Tensor":
        assert shape == ()

        return cls(0, dtype)

    @classmethod
    def create(cls, list_: list, dtype: np.dtype, ctx: dict | None = None) -> "Tensor":
        assert len(list_) == 1

        return cls(list_[0], dtype)

    def reshape(
        self, shape: tuple[int, ...] | int, ctx: dict | None = None
    ) -> "Tensor":
        if shape != ():
            raise ValueError(f"Cannot reshape scalar to {shape!r}")

        return self

    def cast(self, dtype: np.dtype, ctx: dict | None = None) -> "Tensor":
        if dtype == self.dtype:
            return self

        if not np.can_cast(self.dtype, dtype, casting="safe"):
            raise TypeError(f"Cannot cast {self.dtype!r} to {dtype!r}")

        return Scalar(self.value, dtype)
