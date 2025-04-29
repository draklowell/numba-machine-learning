from abc import ABC, abstractmethod

import numpy as np

from nml.device import Device
from nml.parameter import Parameter
from nml.tensor import Tensor
from nml.units import Unit, UnitWithWeights


class DeferredResults(ABC):
    """
    Base class for deferred results in the NML framework.
    This class is created to support parallel inference for multiple models.
    """

    @abstractmethod
    def wait(self) -> Tensor:
        """
        Wait for the deferred results to complete.

        Returns:
            The result of the inference.
        """


class Model:
    """
    Attributes:
        units: The list of units in the model.
        input_shape: The shape of the input tensor (without batch axis).
        input_dtype: The data type of the input tensor.
        output_shape: The shape of the output tensor (without batch axis).
        output_dtype: The data type of the output tensor.
    """

    device: Device
    units: list[Unit]
    input_shape: tuple[int, ...]
    input_dtype: np.dtype
    output_shape: tuple[int, ...]
    output_dtype: np.dtype

    def __init__(
        self,
        units: list[Unit],
        input_shape: tuple[int, ...],
        input_dtype: np.dtype,
        output_shape: tuple[int, ...],
        output_dtype: np.dtype,
    ):
        super().__init__()
        self.units = units
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.output_shape = output_shape
        self.output_dtype = output_dtype

    def get_weights(self) -> dict[str, dict[str, Tensor]]:
        """
        Get the weights of the model.

        Returns:
            Dictionary of weights for each layer.
        """
        return {
            unit.name: unit.get_weights()
            for unit in self.units
            if isinstance(unit, UnitWithWeights)
        }

    def get_parameters(self) -> "dict[str, dict[str, Parameter]]":
        """
        Get the parameters of the model.

        Returns:
            Dictionary of parameters for each layer.
        """
        return {
            unit.name: unit.get_parameters()
            for unit in self.units
            if isinstance(unit, UnitWithWeights)
        }

    def replace_weights(self, weights: dict[str, dict[str, Tensor]]) -> None:
        """
        Replace the weights of the model.

        Args:
            weights: Dictionary of weights for each layer.
        """
        marked = set()
        for unit in self.units:
            if isinstance(unit, UnitWithWeights) and unit.name in weights:
                unit.replace_weights(weights[unit.name])
                marked.add(unit.name)

        for name in weights:
            if name not in marked:
                raise ValueError(f"Layer {name!r} not found in model")

    def __call__(self, tensor: Tensor) -> DeferredResults:
        """
        Call the model with the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            The output tensor.
        """
        return self.infer(tensor)

    def infer(self, tensor: Tensor) -> DeferredResults:
        """
        Infer the output of the model for the given input.

        Args:
            tensor: The input tensor.

        Returns:
            The output tensor.
        """
        if tensor.dtype != self.input_dtype:
            raise TypeError(
                f"Input dtype {tensor.dtype} does not match expected dtype {self.input_dtype}"
            )
        if tensor.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input shape {tensor.shape[1:]} does not match expected shape {self.input_shape}"
            )

        if tensor.device != self.device:
            raise ValueError(
                f"Input device {tensor.device} does not match model device {self.device}"
            )

        return self._infer(tensor)

    @abstractmethod
    def _infer(self, tensor: Tensor) -> DeferredResults:
        pass
