from enum import Enum


class Device(Enum):
    """
    Enum representing the device type for model inference.
    """

    CPU = "cpu"
    GPU = "gpu"
