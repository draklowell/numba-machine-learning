import numpy as np

from nml.cpu import CPUTensor
from nml.device import Device
from nml.tensor import Tensor

try:
    from nml.gpu import GPUTensor
except ImportError:
    GPUTensor = None


class Parameter:
    """
    Parameter class for defining a tensor parameter with specific properties.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    low: np.number | None
    high: np.number | None

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        low: np.number | None = None,
        high: np.number | None = None,
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
                    f"High value {high} exceeds maximum for dtype {dtype}: "
                    f"{np.iinfo(dtype).max + 1}"
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

        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high

    def check_bounds(self, min: np.number, max: np.number) -> bool:
        """
        Check if the parameter value is within the specified bounds.
        Args:
            min: Minimum value (inclusive).
            max: Maximum value (exclusive).
        Returns:
            True if the value is within bounds, False otherwise.
        """
        if self.low is not None and self.low > min:
            return False
        if self.high is not None and self.high <= max:
            return False

        return True

    def create_tensor(self, device: Device) -> Tensor:
        """
        Create a holder for the parameter value.
        Args:
            device: Device to create the holder on.
        Returns:
            Holder for the parameter value.
        """
        match device:
            case Device.CPU:
                return CPUTensor.empty(self.shape, self.dtype)
            case Device.GPU if GPUTensor is not None:
                return GPUTensor.empty(self.shape, self.dtype)
            case _:
                raise NotImplementedError(
                    f"Device {device} is not supported for Parameter."
                )

    def cast(self, tensor: Tensor) -> Tensor:
        """
        Cast the parameter value to the correct type.
        Args:
            value: Value to be casted.
        Returns:
            Casted value.
        """
        if tensor.shape != self.shape:
            raise ValueError(
                f"Value shape {tensor.shape!r} does not match expected shape {self.shape!r}"
            )

        if not np.can_cast(tensor.dtype, self.dtype, casting="safe"):
            raise TypeError(f"Unsupported value dtype: {tensor.dtype!r}")

        return tensor.cast(self.dtype)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.name!r}, "
            f"shape={self.shape!r}, dtype={self.dtype!r}, "
            f"low={self.low!r}, high={self.high!r})"
        )
