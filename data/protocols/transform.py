from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Transform(Protocol):
    def __call__(self, x: NDArray) -> NDArray: ...
