from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


class Parameter(ABC):
    """
    Base class for all parameters in the NML framework.
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def cast(self, value: Any) -> Any:
        """
        Cast the parameter value to the correct type.
        """

    @abstractmethod
    def create(self) -> "ParameterHolder[Any, Parameter]":
        """
        Create a ParameterHolder for this parameter.
        """

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


T = TypeVar("T")
P = TypeVar("P", bound=Parameter)


class ParameterHolder(Generic[T, P]):
    """
    A holder for a parameter value.
    This class is used to encapsulate the parameter value and its validation.
    """

    parameter: P
    _value: T

    def __init__(self, parameter: P):
        self.parameter = parameter
        self._value = None

    def set(self, value: T):
        """
        Validate the parameter value and set it.
        """
        self._value = self.parameter.cast(value)

    def get(self) -> T:
        """
        Get the parameter value.
        """
        return self._value

    def __repr__(self):
        return (
            f"{type(self).__name__}[{type(self.parameter).__name__}]({self._value!r})"
        )
