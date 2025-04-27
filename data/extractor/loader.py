from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from data.protocols.loader import Loader
from data.protocols.transform import Transform


class NumpyLoader(Loader):

    def __init__(self, path: Path, mmap_mode: str | None = "r"):
        self.path = Path(path)
        self.mmap_mode = mmap_mode
        self._data: NDArray = None

    def load(self) -> NDArray:
        if self._data is None:
            self._data = np.load(str(self.path), mmap_mode=self.mmap_mode)
        return self._data

    def __len__(self) -> int:
        return self.load().shape[0]
