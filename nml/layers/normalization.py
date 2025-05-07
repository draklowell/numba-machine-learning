import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import NormalizeUnit


class Normalize(Layer):
    """
    Normalize layer that normalizes the input tensor to a specified range.
    """

    name: str = "normalize"

    def __init__(self, old_min: float, old_max: float, new_min: float, new_max: float):
        self.k = (new_max - new_min) / (old_max - old_min)
        self.b = new_min - self.k * old_min

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> NormalizeUnit:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Expected dtype to be a floating type, got {dtype} instead."
            )

        return NormalizeUnit(
            name,
            shape,
            dtype,
            device,
            self.k,
            self.b,
        )
