from abc import ABC, abstractmethod

import numpy as np

from nml.device import Device
from nml.parameter import Parameter
from nml.tensor import Tensor


class Unit(ABC):
    """
    Base class for all units in the NML framework.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    device: Device

    def __init__(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype, device: Device
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor: Tensor, ctx: dict) -> Tensor:
        """
        Apply the unit to the input tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Output tensor.
        """
        if tensor.device != self.device:
            raise ValueError(
                f"Tensor device {tensor.device} does not "
                f"match unit device {self.device}"
            )

        output = self.infer(tensor, ctx=ctx)
        if output.shape[1:] != self.shape:
            raise ValueError(
                f"Output shape {output.shape} does not match unit shape {self.shape}"
            )

        if output.dtype != self.dtype:
            raise TypeError(
                f"Output dtype {output.dtype} does not match unit dtype {self.dtype}"
            )

        return output

    @abstractmethod
    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        """
        Apply the unit to the input tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Output tensor.
        """


class UnitWithWeights(Unit):
    """
    A unit that has weights.
    """

    _parameters: dict[str, Parameter]
    _weights: dict[str, Tensor]

    def __init__(
        self,
        parameters: list[Parameter],
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
    ):
        super().__init__(name, shape, dtype, device)
        self._parameters = {parameter.name: parameter for parameter in parameters}
        self._weights = {
            parameter.name: parameter.create_tensor(device) for parameter in parameters
        }

    def get_weights(self) -> dict[str, Tensor]:
        """
        Get the weights of the layer.

        Does not copy tensors, so you can modify them in place.

        Returns:
            Dictionary of weights.
        """
        return self._weights.copy()

    def get_parameters(self) -> list[Parameter]:
        """
        Get the configuration of the weights.

        Returns:
            Dictionary of weights configurations.
        """
        return list(self._parameters.values())

    def replace_weights(self, weights: dict[str, Tensor]):
        """
        Replace the weights of the layer.

        Args:
            weights: Dictionary of weights to set.
        """
        for name, tensor in weights.items():
            if name not in self._weights:
                raise ValueError(f"Parameter {name!r} does not exist")

            self._weights[name] = self._parameters[name].cast(tensor)
