from enum import Enum

import numpy as np
from numba import cuda
from data.extractor.prefetch import PrefetchSampler
from data.protocols.loader import Loader
from data.protocols.transform import Transform
from data.extractor.sampler import GPUSampler, IndexShuffleSampler


class StorageDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"


class DataManager:

    def __init__(
        self,
        loader: Loader,
        transform: Transform = None,
        sampler_seed: int = None,
        prefetch: bool = False,
        prefetch_queue: int = 2,
        storage_device: StorageDevice = StorageDevice.CPU,
        return_device: StorageDevice = StorageDevice.CPU,
    ):
        self.loader = loader
        self.transform = transform
        self.sampler_seed = sampler_seed
        self.prefetch = prefetch
        self.prefetch_queue = prefetch_queue
        self.storage_device = storage_device
        self.return_device = return_device

        if storage_device == StorageDevice.CUDA and not cuda.is_available():
            raise RuntimeError("CUDA is not available. Use CPU instead.")
        if (
            self.return_device == StorageDevice.CUDA
            and self.storage_device != StorageDevice.CUDA
        ):
            raise ValueError(
                "Cannot return CUDA data if it was not loaded on CUDA device."
            )

        self._sampler = None

    def prepare(self) -> None:
        raw = self.loader.load()

        if self.storage_device == StorageDevice.CPU:
            data = (
                raw
                if raw.flags["WRITEABLE"] and raw.flags["C_CONTIGUOUS"]
                else np.ascontiguousarray(raw)
            )

            if self.transform is not None:
                data = self.transform(data)
            sampler = IndexShuffleSampler(data, self.sampler_seed)

        elif self.storage_device == StorageDevice.CUDA:

            data_cpu = np.ascontiguousarray(raw)
            d_raw = cuda.to_device(data_cpu)

            if self.transform is not None:
                d_data = self.transform(d_raw)
            else:
                d_data = d_raw

            sampler = GPUSampler(d_data, self.sampler_seed)

        self._sampler = (
            PrefetchSampler(sampler, self.prefetch_queue) if self.prefetch else sampler
        )

    def get_samples(self, batch_size: int) -> np.ndarray:
        if self._sampler is None:
            raise RuntimeError("DataManager not prepared. Call prepare() first.")

        batch = self._sampler.get_samples(batch_size)

        if (
            self.storage_device == StorageDevice.CUDA
            and self.return_device == StorageDevice.CPU
        ):
            return batch.copy_to_host()
        return batch

    def __len__(self) -> int:
        return len(self.loader)
