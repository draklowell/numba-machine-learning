from typing import Any

from nml.layers import Layer
from nml.parameters import Parameter


class SequentialModel:
    """
    A simple sequential model for building neural networks.

    Args:
        layers: A list of layers to be added to the model.
    """

    def __init__(self, *layers: Layer):
        names = set()
        for layer in layers:
            if layer.name in names:
                raise ValueError(f"Layer name {layer.name!r} already exists")
            names.add(layer.name)

        self.layers = layers
        # self._built = {}

    def get_weights(self) -> dict[str, dict[str, Any]]:
        """
        Get the weights of the model.

        Returns:
            Dictionary of weights for each layer.
        """
        return {layer.name: layer.get_weights() for layer in self.layers}

    def get_parameters(self) -> dict[str, dict[str, Parameter]]:
        """
        Get the parameters of the model.

        Returns:
            Dictionary of parameters for each layer.
        """
        return {layer.name: layer.get_parameters() for layer in self.layers}

    def set_weights(self, weights: dict[str, dict[str, Any]]) -> None:
        """weights = {}
        for name, layer in self._iterate_layers():
            weights[name] = layer.get_weights()

        return weights

        Set the weights of the model.

        Args:
            weights: Dictionary of weights for each layer.
        """
        marked = set()
        for layer in self.layers:
            if layer.name in weights:
                layer.set_weights(weights[layer.name])
                marked.add(layer.name)
            else:
                raise ValueError(
                    f"Layer {layer.name!r} not found in weights dictionary"
                )

        for name in weights:
            if name not in marked:
                raise ValueError(f"Layer {name!r} not found in model")

    def update_weights(self, weights: dict[str, dict[str, Any]]) -> None:
        """
        Update the weights of the model.

        Args:
            weights: Dictionary of weights for each layer.
        """
        marked = set()
        for layer in self.layers:
            if layer.name in weights:
                layer.update_weights(weights[layer.name])
                marked.add(layer.name)

        for name in weights:
            if name not in marked:
                raise ValueError(f"Layer {name!r} not found in model")

    def infer(self, x: Any) -> Any:
        """
        Infer the output of the model for the given input.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        for layer in self.layers:
            x = layer.infer(x)

        return x
