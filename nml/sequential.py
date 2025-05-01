from dataclasses import dataclass

import numpy as np

from nml.cpu_model import CPUModel
from nml.device import Device
from nml.layers import Layer
from nml.model import Model

try:
    from nml.gpu_model import GPUModel
except ImportError:
    GPUModel = None


@dataclass
class Input:
    """
    Input layer for defining the input shape and data type.
    """

    shape: tuple[int, ...]
    dtype: np.dtype


class Sequential:
    """
    Sequential model for building a neural network.

    Args:
        layers: A list of layers to be added to the model.
    """

    def __init__(self, *layers: Layer):
        if not isinstance(layers[0], Input):
            raise TypeError("First layer must be an Input layer")

        self.input = layers[0]
        self.layers = layers[1:]

    def build(self, device: Device = Device.CPU) -> Model:
        """
        Build the model by creating units for each layer.

        Args:
            device: The device to use for the model.

        Returns:
            A Model object representing the built model."""
        units = []
        shape = self.input.shape
        dtype = self.input.dtype
        for idx, layer in enumerate(self.layers, start=1):
            unit = layer(shape, dtype, f"{layer.name}-{idx}", device)
            units.append(unit)
            shape = unit.shape
            dtype = unit.dtype

        match device:
            case Device.CPU:
                return CPUModel(
                    units=units,
                    input_shape=self.input.shape,
                    input_dtype=self.input.dtype,
                    output_shape=shape,
                    output_dtype=dtype,
                )
            case Device.GPU if GPUModel is not None:
                return GPUModel(
                    units=units,
                    input_shape=self.input.shape,
                    input_dtype=self.input.dtype,
                    output_shape=shape,
                    output_dtype=dtype,
                )

        raise NotImplementedError(
            f"Device {device} is not supported by the sequential model"
        )
