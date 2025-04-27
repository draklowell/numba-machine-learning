import numpy as np
from numba import cuda
from numpy.typing import NDArray

try:
    GPU_AVAILABLE = cuda.is_available()
except Exception:
    GPU_AVAILABLE = False


class IndexShuffleSampler:
    """
    Samples without replacement by pre-shuflling indices and slicing
    """

    def __init__(self, data: NDArray, seed: int | None = None):
        self.data = data
        self.N = data.shape[0]
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.N, dtype=np.int64)
        self._shufle()
        self.ptr = 0

    def _shuffle(self) -> None:
        self.rng.shuffle(self.indices)
        self.ptr = 0

    def get_samples(self, batch_size: int) -> NDArray:
        if batch_size > self.N:
            raise ValueError("Exceeded dataset size")
        if self.ptr + batch_size > self.N:
            part1 = self.indices[self.ptr :]
            self._shuffle()
            part2 = self.indices[: batch_size - part1.size]
            self.ptr = part2.size
            batch_idx = np.concatenate([part1, part2])
        else:
            batch_idx = self.indices[self.ptr : self.ptr + batch_size]
            self.ptr += batch_size
        return self.data[batch_idx]


class GPUSampler:
    """
    Samples from a GPU-resident array by shuffling host indices and indexing device data.
    """

    def __init__(self, d_data, seed: int | None = None):
        if not GPU_AVAILABLE:
            raise RuntimeError("No cuda detected")
        self.d_data = d_data
        self.N = d_data.size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.N, dtype=np.int64)
        self._shuffle()
        self.ptr = 0

    def _shuffle(self):
        self.rng.shuffle(self.indices)
        self.ptr = 0

    def get_samples(self, batch_size: int):
        if batch_size > self.N:
            raise ValueError("Exceeded dataset size")
        if self.ptr + batch_size > self.N:
            part1 = self.indices[self.ptr :]
            self._shuffle()
            part2 = self.indices[: batch_size - part1.size]
            self.ptr = part2.size
            batch_idx = np.concatenate([part1, part2])
        else:
            batch_idx = self.indices[self.ptr : self.ptr + batch_size]
            self.ptr += batch_size
        return self.d_data[batch_idx]
