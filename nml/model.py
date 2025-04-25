from collections import Counter
from typing import Any, Generator

from nml.layers import Layer
from nml.parameters import Parameter


class SequentialModel:
    """
    A simple sequential model for building neural networks.

    Args:
        layers: A list of layers to be added to the model.
    """

    def __init__(self, *layers: Layer):
        counter = Counter()
        for layer in layers:
            layer.include(counter[layer.name])
            counter[layer.name] += 1

        self.layers = layers

    def _iterate_layers(self) -> Generator[tuple[str, Layer], None, None]:
        for layer in self.layers:
            if layer.idx:
                yield f"{layer.name}_{layer.idx}", layer
            else:
                yield layer.name, layer

    def get_weights(self) -> dict[str, dict[str, Any]]:
        """
        Get the weights of the model.

        Returns:
            Dictionary of weights for each layer.
        """
        return {name: layer.get_weights() for name, layer in self._iterate_layers()}

    def get_parameters(self) -> dict[str, dict[str, Parameter]]:
        """
        Get the parameters of the model.

        Returns:
            Dictionary of parameters for each layer.
        """
        return {name: layer.get_parameters() for name, layer in self._iterate_layers()}

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
        for name, layer in self._iterate_layers():
            if name in weights:
                layer.set_weights(weights[name])
                marked.add(name)
            else:
                raise ValueError(f"Layer {name!r} not found in weights dictionary")

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
        for name, layer in self._iterate_layers():
            if name in weights:
                layer.update_weights(weights[name])
                marked.add(name)

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
