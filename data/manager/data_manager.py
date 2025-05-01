from enum import Enum
from typing import Union

import numpy as np
from numba import cuda

from data.core.quantize_cpu import CPUStateDownSampler
from data.core.quantize_gpu import CUDAStateDownSampler


class Device(Enum):
    """Device specification for processing or storing data"""

    CPU = "cpu"
    GPU = "gpu"


class DataManager:
    """
    Data manager for MNIST digits that handles different processing and memory modes.
    Supports loading, quantization (downsampling), and batch sampling of MNIST data.
    """

    def __init__(
        self,
        data_path: str,
        states: int,
        batch_size: int,
        process_device: Device = Device.CPU,
        storage_device: Device = Device.CPU,
    ):
        self.data_path = data_path
        self.bit_width = states
        self.batch_size = batch_size
        self.process_device = process_device
        self.storage_device = storage_device

        self.data_cpu = None
        self.data_gpu = None

        if self.process_device == Device.CPU and self.storage_device == Device.GPU:
            raise RuntimeError(
                "Only three modes available: 'cpu to cpu', 'gpu_to_cpu', 'gpu_to_gpu'"
            )
        if self.process_device == Device.CPU:
            self.downsampler = CPUStateDownSampler(self.bit_width)
        elif self.process_device == Device.GPU:
            if not cuda.is_available():
                raise RuntimeError(f"CUDA is not available for GPU processing")
            self.downsampler = CUDAStateDownSampler(self.bit_width)

        self.all_indices = None

    def load_data(self) -> None:
        """Load MNIST data from disk into CPU memory."""
        self.data_cpu = np.load(self.data_path)

        if (
            len(self.data_cpu.shape) != 3
            or self.data_cpu.shape[1:] != (28, 28)
            or self.data_cpu.dtype != np.uint8
        ):
            raise ValueError(
                f"Expected MNIST data of shape (N, 28, 28) and dtype uint8, "
                f"got shape {self.data_cpu.shape} and dtype {self.data_cpu.dtype}"
            )
        self.all_indices = np.arange(self.data_cpu.shape[0])

    def downsample(self) -> None:
        """Apply quantization based on the selected processing and storage devices."""
        if self.data_cpu is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if self.process_device == Device.CPU:
            self.data_cpu = self.downsampler(self.data_cpu)
            if self.storage_device == Device.GPU:
                self.data_gpu = cuda.to_device(self.data_cpu)
                self.data_cpu = None

        elif self.process_device == Device.GPU:
            self.data_gpu = cuda.to_device(self.data_cpu)
            self.downsampler(self.data_gpu)
            if self.storage_device == Device.CPU:
                self.data_cpu = self.data_gpu.copy_to_host()
            else:
                self.data_cpu = None

    def get_samples(self) -> Union[np.ndarray, "cuda.devicearray.DeviceNDArray"]:
        """
        Randomly select batch_size images from the quantized tensor.

        Returns:
            For CPU storage: NumPy ndarray of shape (batch_size, 28, 28)
            For GPU storage: CUDA device array of shape (batch_size, 28, 28)

            In all cases, the array has dtype=uint8 with values in range [0, 2^bit_width-1]
        """
        indices = np.random.randint(0, len(self.all_indices), size=self.batch_size)

        if self.storage_device == Device.CPU:
            if self.data_cpu is None:
                raise RuntimeError(
                    "CPU data not available. Ensure downsample() has been called."
                )
            return self.data_cpu[indices]
        else:
            if self.data_gpu is None:
                raise RuntimeError(
                    "GPU data not available. Ensure downsample() has been called."
                )
            data_cpu = self.data_gpu.copy_to_host()
            batch_cpu = data_cpu[indices]
            return cuda.to_device(batch_cpu)
