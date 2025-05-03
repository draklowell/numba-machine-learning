import numpy as np
from numba import cuda

from loader.core.quantize_cpu import CPUStateDownSampler
from loader.core.quantize_gpu import CUDAStateDownSampler
from nml import Device
from nml.cpu.tensor import CPUTensor
from nml.gpu.tensor import GPUTensor
from nml.tensor import Tensor


class DataManager:
    """
    Data manager for MNIST digits that handles different processing and memory modes.
    Supports loading, quantization (downsampling), and batch sampling of MNIST data.

    Returns proper tensor objects (CPUTensor, GPUTensor) instead of raw arrays.
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
        """Load MNIST data from disk into CPU memory as CPUTensor."""
        data_array = np.load(self.data_path)

        if (
            len(data_array.shape) != 3
            or data_array.shape[1:] != (28, 28)
            or data_array.dtype != np.uint8
        ):
            raise ValueError(
                f"Expected MNIST data of shape (N, 28, 28) and dtype uint8, "
                f"got shape {data_array.shape} and dtype {data_array.dtype}"
            )

        self.data_cpu = CPUTensor(data_array)
        self.all_indices = np.arange(self.data_cpu.shape[0])

    def downsample(self) -> None:
        """Apply quantization based on the selected processing and storage devices."""
        if self.data_cpu is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if self.process_device == Device.CPU:
            raw_data = self.downsampler(self.data_cpu.array)
            self.data_cpu = CPUTensor(raw_data)

            if self.storage_device == Device.GPU:
                self.data_gpu = GPUTensor(cuda.to_device(self.data_cpu.array))
                self.data_cpu = None

        elif self.process_device == Device.GPU:
            d_array = cuda.to_device(self.data_cpu.array)
            self.downsampler(d_array)

            if self.storage_device == Device.CPU:
                self.data_cpu = CPUTensor(d_array.copy_to_host())
                self.data_gpu = None
            else:
                self.data_gpu = GPUTensor(d_array)
                self.data_cpu = None

    def get_samples(self) -> Tensor:
        """
        Randomly select batch_size images from the quantized tensor.

        Returns:
            CPUTensor or GPUTensor of shape (batch_size, 28, 28) with dtype=uint8
            and values in range [0, 2^bit_width-1]
        """
        indices = np.random.randint(0, len(self.all_indices), size=self.batch_size)

        if self.storage_device == Device.CPU:
            if self.data_cpu is None:
                raise RuntimeError(
                    "CPU data not available. Ensure downsample() has been called."
                )
            return CPUTensor(self.data_cpu.array[indices])
        else:
            if self.data_gpu is None:
                raise RuntimeError(
                    "GPU data not available. Ensure downsample() has been called."
                )
            data_cpu = self.data_gpu.array.copy_to_host()
            batch_cpu = data_cpu[indices]
            return GPUTensor(cuda.to_device(batch_cpu))
