from abc import ABC, abstractmethod
from typing import Any

from nml.parameters import Parameter, ParameterHolder


class Layer(ABC):
    """
    Base class for all layers in the NML framework.
    """

    name: str
    _parameters: dict[str, ParameterHolder]

    def __init__(self, name: str):
        self.name = name
        self._parameters = {}

    def _create_parameter(self, parameter: Parameter):
        """
        Create parameters for the layer.

        Args:
            parameter: Parameter to create.
        """
        if parameter.name in self._parameters:
            raise ValueError(
                f"Parameter with the name {parameter.name!r} already exists"
            )

        self._parameters[parameter.name] = parameter.create()

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
    def infer(self, x: Any) -> Any:
        """
        Apply the layer to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
