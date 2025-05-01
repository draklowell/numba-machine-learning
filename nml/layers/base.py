from abc import ABC, abstractmethod

import numpy as np

from nml.device import Device
from nml.units import Unit


class Layer(ABC):
    """
    Base class for all layers in the NML framework.
    """

    name: str

    @abstractmethod
    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> Unit:
        pass
