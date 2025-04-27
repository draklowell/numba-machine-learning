from enum import Enum

import numpy as np
from numba import cuda
from prefetch import PrefetchSampler
from protocols.loader import Loader
from protocols.transform import Transform
from sampler import GPUSampler, IndexShuffleSampler


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
        # Call the loader method immediately
        raw = self.loader.load()  # Note the added parentheses!

        if self.storage_device == StorageDevice.CPU:
            # Ensure the array is contiguous
            # Only copy if needed (if already writeable we can avoid an extra copy)
            data = (
                raw
                if raw.flags["WRITEABLE"] and raw.flags["C_CONTIGUOUS"]
                else np.ascontiguousarray(raw)
            )

            # Apply transformation in place if possible to reduce memory overhead.
            if self.transform is not None:
                data = self.transform(data)

            # Create the CPU sampler using the transformed data
            sampler = IndexShuffleSampler(data, self.sampler_seed)

        elif self.storage_device == StorageDevice.CUDA:
            # Optionally: allocate host memory pinned for faster transfer.
            # For example, if raw is not already pinned, you could use:
            # pinned_data = cuda.pinned_array(raw.shape, raw.dtype)
            # np.copyto(pinned_data, raw)
            # d_raw = cuda.to_device(pinned_data, stream=cuda.stream() )
            # For simplicity we directly copy to device here.
            data_cpu = np.ascontiguousarray(raw)
            d_raw = cuda.to_device(data_cpu)

            # If a GPU transform is provided, it should operate on device arrays.
            if self.transform is not None:
                d_data = self.transform(d_raw)
            else:
                d_data = d_raw

            # Create the GPU sampler using the device data
            sampler = GPUSampler(d_data, self.sampler_seed)

        # Wrap the sampler with a prefetcher if needed
        self._sampler = (
            PrefetchSampler(sampler, self.prefetch_queue) if self.prefetch else sampler
        )

    def get_samples(self, batch_size: int) -> np.ndarray:
        if self._sampler is None:
            raise RuntimeError("DataManager not prepared. Call prepare() first.")

        batch = self._sampler.get_samples(batch_size)

        # If the data are stored on the GPU but should be returned on the CPU,
        # perform an asynchronous copy if you also use streams (this example uses a direct copy).
        if (
            self.storage_device == StorageDevice.CUDA
            and self.return_device == StorageDevice.CPU
        ):
            return (
                batch.copy_to_host()
            )  # Could be replaced with an async copy using streams.
        return batch

    def __len__(self) -> int:
        return len(self.loader)
