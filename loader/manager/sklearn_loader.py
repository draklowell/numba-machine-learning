from typing import Optional, Tuple

import numpy as np
from skimage.transform import resize
from sklearn.datasets import load_digits

from nml import CPUTensor, Device
from nml.tensor import Tensor

try:
    from numba import cuda

    from nml import GPUTensor
except ImportError:
    cuda = None
    GPUTensor = None


class SklearnBalancedDataLoader:
    """
    A data loader for the sklearn digits dataset that provides balanced batches.

    This loader ensures that each batch contains at least one sample from each class (0-9),
    if available in the dataset, and fills the rest of the batch with random samples.
    Optionally resizes the 8x8 digit images to a specified size.
    """

    def __init__(
        self,
        batch_size: int,
        resize_to: Optional[Tuple[int, int]] = None,
        random_state: Optional[int] = None,
        process_device: Device = Device.CPU,
        storage_device: Device = Device.CPU,
    ):
        """
        Initialize the loader with the sklearn digits dataset.

        Args:
            batch_size: Number of samples per batch.
            resize_to: Optional tuple (height, width) to resize the images.
                       Default is None (keep original 8x8 size).
            random_state: Optional seed for random number generation.
            process_device: Device to use for processing (CPU or GPU).
            storage_device: Device to store the data (CPU or GPU).
        """
        self.batch_size = batch_size
        self.process_device = process_device
        self.storage_device = storage_device
        if self.process_device == Device.CPU and self.storage_device == Device.GPU:
            raise NotImplementedError(
                "Only three modes available: 'cpu to cpu', 'gpu_to_cpu', 'gpu_to_gpu'"
            )

        if self.storage_device == Device.GPU and GPUTensor is None:
            raise NotImplementedError(
                "GPU storage is not supported. Please install the GPU version of NML."
            )

        if self.storage_device not in {Device.CPU, Device.GPU}:
            raise NotImplementedError(
                f"Storage device {self.storage_device} not supported."
            )

        self.X_cpu = None
        self.X_gpu = None
        self.y_cpu = None
        self.y_one_hot_cpu = None

        digits = load_digits(return_X_y=False)
        X = digits.images
        y = digits.target

        # Ensure correct data types - convert from float64 to uint8
        # sklearn's digits dataset has values 0-16, scale to 0-255 for uint8
        X = np.round(X * (255.0 / 16.0)).astype(np.uint8)

        if resize_to is not None:
            resized_images = np.zeros((len(X),) + resize_to, dtype=np.uint8)
            for i, img in enumerate(X):
                resized = resize(img, resize_to, preserve_range=True).astype(np.uint8)
                resized_images[i] = resized
            X = resized_images

        y_one_hot = np.zeros((len(y), 10), dtype=np.uint8)
        y_one_hot[np.arange(len(y)), y] = 1
        if self.storage_device == Device.CPU:
            self.X_cpu = X
            self.y_cpu = y
            self.y_one_hot_cpu = y_one_hot
        else:
            self.X_gpu = cuda.to_device(X)
            self.y_cpu = y
            self.y_one_hot_cpu = y_one_hot

        self.rng = np.random.RandomState(random_state)
        self.indices_by_class = {c: np.where(y == c)[0] for c in range(10)}

    def _get_balanced_indices(self) -> np.ndarray:
        """
        Generate a balanced batch of indices with at least one sample per class.

        Returns:
            Array of selected indices with shape (batch_size,)
        """
        selected_indices = []
        for c in range(10):
            if len(self.indices_by_class[c]) > 0:
                idx = self.rng.choice(self.indices_by_class[c], 1)[0]
                selected_indices.append(idx)

        if self.batch_size > len(selected_indices):
            remaining_indices = np.setdiff1d(
                np.arange(len(self.y_cpu)), selected_indices
            )

            if len(remaining_indices) > 0:
                additional_count = min(
                    self.batch_size - len(selected_indices), len(remaining_indices)
                )
                additional_indices = self.rng.choice(
                    remaining_indices, additional_count, replace=False
                )
                selected_indices.extend(additional_indices)

        self.rng.shuffle(selected_indices)
        return np.array(selected_indices[: self.batch_size])

    def get_samples(self) -> Tuple[Tensor, Tensor]:
        """
        Get a balanced batch of samples, matching DataManager's behavior.

        Returns:
            Tuple containing:
                - X_batch: Tensor of images with shape (batch_size, height, width) on storage device
                - y_batch: Tensor of one-hot encoded labels with shape (batch_size, 10) on CPU
        """
        selected_indices = self._get_balanced_indices()

        y_batch = self.y_one_hot_cpu[selected_indices]
        batch_labels = CPUTensor(y_batch)

        if self.storage_device == Device.CPU:
            if self.X_cpu is None:
                raise RuntimeError("CPU data not available")
            # Ensure uint8 data type
            return CPUTensor(self.X_cpu[selected_indices].astype(np.uint8)), batch_labels
        else:
            if self.X_gpu is None:
                raise RuntimeError("GPU data not available")
            data_cpu = self.X_gpu.copy_to_host()
            batch_cpu = data_cpu[selected_indices].astype(np.uint8)
            return GPUTensor(cuda.to_device(batch_cpu)), batch_labels

    def get_raw_labels(self) -> np.ndarray:
        """
        Get the raw class labels for the entire dataset.

        Returns:
            Array of class labels with shape (n_samples,)
        """
        return self.y_cpu.copy()

    def __call__(self) -> Tuple[Tensor, Tensor]:
        """
        Get a balanced batch of samples.

        Returns:
            Tuple containing:
                - X_batch: Tensor of images with shape (batch_size, height, width) on storage device
                - y_batch: Tensor of one-hot encoded labels with shape (batch_size, 10) on CPU
        """
        return self.get_samples()
