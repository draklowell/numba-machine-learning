from typing import Any

import numpy as np

from nml.layers import InferableLayer, Layer
from nml.models.base import InferableModel, Model
from nml.parameters import Parameter


class Input:
    """
    A class representing the input layer of a model.

    Attributes:
        shape: The shape of the input tensor.
        dtype: The data type of the input tensor.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype):
        self.shape = shape
        self.dtype = dtype


class InferableSequential(InferableModel):
    """
    A class representing an inferable sequential model.

    Attributes:
        model: The model used for inference.
        input_shape: The shape of the input tensor.
        input_dtype: The data type of the input tensor.
        output_shape: The shape of the output tensor.
        output_dtype: The data type of the output tensor.
    """

    layers: list[InferableLayer]
    input_shape: tuple[int, ...]
    input_dtype: np.dtype
    output_shape: tuple[int, ...]
    output_dtype: np.dtype

    def __init__(
        self,
        layers: list[InferableLayer],
        input_shape: tuple[int, ...],
        input_dtype: np.dtype,
        output_shape: tuple[int, ...],
        output_dtype: np.dtype,
    ):
        super().__init__()
        self.layers = layers
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.output_shape = output_shape
        self.output_dtype = output_dtype

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

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Infer the output of the model for the given input.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        if isinstance(x, (int, float, list)):
            x = np.array(x, dtype=self.input_dtype)

        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected input type is np.ndarray, but got {type(x)}")
        if x.dtype != self.input_dtype:
            raise TypeError(
                f"Input dtype {x.dtype} does not match expected dtype {self.input_dtype}"
            )
        if x.shape != self.input_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match expected shape {self.input_shape}"
            )

        for layer in self.layers:
            x = layer.infer(x)

        return x


class Sequential(Model):
    """
    A simple sequential model for building neural networks.

    Args:
        layers: A list of layers to be added to the model.
    """

    def __init__(self, input_layer: Input, *layers: Layer):
        if not isinstance(input_layer, Input):
            raise TypeError("First layer must be an Input layer")

        self.input = input_layer
        self.layers = layers

    def build(self) -> InferableSequential:
        """
        Build the model with the given input shape and data type.

        Args:
            shape: The input shape.
            dtype: The input data type.

        Returns:
            The output shape and data type.
        """
        inferable_layers = []
        shape = self.input.shape
        dtype = self.input.dtype
        for idx, layer in enumerate(self.layers, start=1):
            inferable, shape, dtype = layer.build(idx, shape, dtype)
            inferable_layers.append(inferable)

        return InferableSequential(
            layers=inferable_layers,
            input_shape=self.input.shape,
            input_dtype=self.input.dtype,
            output_shape=shape,
            output_dtype=dtype,
        )
