from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Loader(Protocol):
    def load(self) -> NDArray: ...
    def __len__(self) -> int: ...
