from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from numpy.typing import NDArray

from nml.parameters import Parameter


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"


class DeferredInference(ABC):
    """
    Base class for deferred inference in the NML framework.
    This class is create to support parallel inference for multiple models.
    """

    def wait(self) -> NDArray:
        """
        Wait for the deferred inference to complete.
        This method should be implemented by subclasses to handle the completion of deferred
        inference.
        """


class ReadyToUseInference(DeferredInference):
    """
    Class for ready-to-use inference in the NML framework.
    This class is used to handle the result of inference without deferring.
    """

    def __init__(self, result: NDArray) -> None:
        self.result = result

    def wait(self) -> NDArray:
        """
        Wait for the deferred inference to complete and return the result.
        This method returns the result of the inference.
        """
        return self.result


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
    def set_weights(
        self, weights: dict[str, dict[str, Any]], update: bool = False
    ) -> None:
        """
        Set the weights of the model.
        This method should be implemented by subclasses to set the model's weights.
        """

    @abstractmethod
    def infer(self, x: NDArray, device: Device = Device.CPU) -> DeferredInference:
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
