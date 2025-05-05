from enum import Enum, auto


class Device(Enum):
    """
    Enum representing the device type for model inference.
    """

    CPU = auto()
    GPU = auto()
