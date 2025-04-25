import numpy as np

from nml.parameters.base import Parameter, ParameterHolder


class TensorParameter(Parameter):
    """
    A parameter representing a tensor.
    This class is used to encapsulate the tensor value and its validation.
    """

    def __init__(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype, low=None, high=None
    ):
        """
        Initialize the TensorParameter.

        Args:
            name: Name of the parameter.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            low: Minimum value for the tensor (inclusive).
            high: Maximum value for the tensor (exclusive).
        """
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.integer):
            if high is not None and high > np.iinfo(dtype).max + 1:
                raise ValueError(
                    f"High value {high} exceeds maximum for dtype {dtype}: {np.iinfo(dtype).max + 1}"
                )

            if low is not None and low < np.iinfo(dtype).min:
                raise ValueError(
                    f"Low value {low} is less than minimum for dtype {dtype}: {np.iinfo(dtype).min}"
                )
        elif np.issubdtype(dtype, np.floating):
            if high is not None and high > np.finfo(dtype).max:
                raise ValueError(
                    f"High value {high} exceeds maximum for dtype {dtype}: {np.finfo(dtype).max}"
                )

            if low is not None and low < np.finfo(dtype).min:
                raise ValueError(
                    f"Low value {low} is less than minimum for dtype {dtype}: {np.finfo(dtype).min}"
                )
        else:
            raise TypeError(f"Unsupported dtype: {dtype!r}")

        if low is not None and high is not None and low >= high:
            raise ValueError(f"Invalid bounds: [{low}, {high})")

        super().__init__(name)

        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high

    def generate_tenosr(self) -> np.ndarray:
        """
        Generate a tensor with the specified shape and data type.

        Returns:
            Generated tensor.
        """
        if np.issubdtype(self.dtype, np.integer):
            low = self.low
            if low is None:
                low = np.iinfo(self.dtype).min
            high = self.high
            if high is None:
                high = np.iinfo(self.dtype).max + 1

            return np.random.randint(
                low=low,
                high=high,
                size=self.shape,
                dtype=self.dtype,
            )

        if self.low is None and self.high is None:
            return np.random.normal(size=self.shape).astype(self.dtype)

        if self.low is None or self.high is None:
            return np.zeros(self.shape, dtype=self.dtype)

        try:
            return np.random.uniform(
                low=self.low,
                high=self.high,
                size=self.shape,
            ).astype(self.dtype)
        except OverflowError:  # If the range is too large, fallback to zeros
            return np.zeros(self.shape, dtype=self.dtype)

    def create(self) -> ParameterHolder[np.ndarray, Parameter]:
        holder = ParameterHolder(self)
        holder.set(self.generate_tenosr())
        return holder

    def cast(self, value: np.ndarray | int | float) -> np.ndarray:
        """
        Cast the parameter value to the correct type.

        Args:
            value: Value to be casted.

        Returns:
            Casted value.
        """
        if isinstance(value, (int, float)):
            if (self.high is not None and value > self.high) or (
                self.low is not None and value < self.low
            ):
                raise ValueError(
                    f"Value {value!r} is out of bounds [{self.low}, {self.high})"
                )

            value = np.array(value, dtype=self.dtype)

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Unsupported value type: {type(value)}")

        if value.shape != self.shape:
            raise ValueError(
                f"Value shape {value.shape!r} does not match expected shape {self.shape!r}"
            )

        if not np.can_cast(value.dtype, self.dtype, casting="safe"):
            raise TypeError(f"Unsupported value dtype: {value.dtype!r}")

        value = value.astype(self.dtype)

        if self.high is not None and (
            (value.ndim != 0 and max(value) >= self.high)
            or (value.ndim == 0 and value >= self.high)
        ):
            raise ValueError(
                f"Value {value!r} is out of bounds [{self.low}, {self.high})"
            )
        if self.low is not None and (
            (value.ndim != 0 and min(value) < self.low)
            or (value.ndim == 0 and value < self.low)
        ):
            raise ValueError(
                f"Value {value!r} is out of bounds [{self.low}, {self.high})"
            )

        return value

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, shape={self.shape!r}, dtype={self.dtype!r}, low={self.low!r}, high={self.high!r})"
