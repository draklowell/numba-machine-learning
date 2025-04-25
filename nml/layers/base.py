from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from nml.parameters import Parameter, ParameterHolder


class InferableLayer(ABC):
    """
    Base class for all layers in the NML framework that can be inferred.
    """

    name: str
    _parameters: dict[str, ParameterHolder]

    def __init__(self, name: str, parameters: list[Parameter] = None):
        self.name = name
        self._parameters = {
            parameter.name: parameter.create() for parameter in parameters or []
        }

    def _get_parameter(self, name: str) -> Any:
        """
        Get the parameter by name.

        Args:
            name: Name of the parameter.

        Returns:
            The parameter value.
        """
        if name not in self._parameters:
            raise ValueError(f"Parameter {name!r} does not exist")

        return self._parameters[name].get()

    def get_weights(self) -> dict[str, Any]:
        """
        Get the weights of the layer.

        Returns:
            Dictionary of weights.
        """
        return {name: param.get() for name, param in self._parameters.items()}

    def get_parameters(self) -> dict[str, Parameter]:
        """
        Get the configuration of the weights.

        Returns:
            Dictionary of weights configurations.
        """
        return {name: param.parameter for name, param in self._parameters.items()}

    def set_weights(self, weights: dict[str, Any]):
        """
        Set the weights of the layer.

        Args:
            weights: Dictionary of weights to set.
        """
        marked = set()
        for name, value in weights.items():
            if name not in self._parameters:
                raise ValueError(f"Parameter {name!r} does not exist")

            self._parameters[name].set(value)
            marked.add(name)

        for name in self._parameters:
            if name not in marked:
                raise ValueError(f"Parameter {name!r} not found in weights dictionary")

    def update_weights(self, weights: dict[str, Any]):
        """
        Update the weights of the layer.

        Args:
            weights: Dictionary of weights to update.
        """
        for name, value in weights.items():
            if name not in self._parameters:
                raise ValueError(f"Parameter {name!r} does not exist")

            self._parameters[name].set(value)

    @abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the layer to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """


class Layer(ABC):
    """
    Base class for all layers in the NML framework.
    """

    name: str = "layer"

    @abstractmethod
    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableLayer, tuple[int, ...], np.dtype]:
        """
        Build the layer with the given shape and data type.

        Args:
            idx: Index of the layer.
            shape: Shape of the input tensor.
            dtype: Data type of the input tensor.

        Returns:
            Tuple of shape and data type.
        """
