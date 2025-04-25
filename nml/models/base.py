from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from nml.parameters import Parameter


class InferableModel(ABC):
    """
    Base class for all models in the NML framework that can be inferred.
    This class provides a common interface for all models.
    """

    @abstractmethod
    def get_weights(self) -> dict[str, dict[str, Any]]:
        """
        Get the weights of the model.
        This method should be implemented by subclasses to return the model's weights.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, dict[str, Parameter]]:
        """
        Get the parameters of the model.
        This method should be implemented by subclasses to return the model's parameters.
        """

    @abstractmethod
    def set_weights(self, weights: dict[str, dict[str, Any]]) -> None:
        """
        Set the weights of the model.
        This method should be implemented by subclasses to set the model's weights.
        """

    @abstractmethod
    def update_weights(self, weights: dict[str, dict[str, Any]]) -> None:
        """
        Update the weights of the model.
        This method should be implemented by subclasses to update the model's weights.
        """

    @abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input data.
        This method should be implemented by subclasses to perform inference.
        """


class Model(ABC):
    """
    Base class for all models in the NML framework.
    This class provides a common interface for all models.
    """

    @abstractmethod
    def build(self) -> InferableModel:
        """
        Build the model.
        This method should be implemented by subclasses to define the model architecture.
        """
