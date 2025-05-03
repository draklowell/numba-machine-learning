import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from loader.manager.data_manager import DataManager
from loader.manager.downloader import Downloader
from nml import Device


def download_test_data(path: str = "data/datasets/mnist", force: bool = False) -> str:
    """
    Download a test dataset if it doesn't exist

    Args:
        path: Path to store the dataset
        force: Whether to force download even if files exist

    Returns:
        Path to the numpy array file
    """
    os.makedirs(path, exist_ok=True)
    numpy_path = os.path.join(path, "train_images.npy")

    if not os.path.exists(numpy_path) or force:
        print("Downloading MNIST dataset...")
        downloader = Downloader(path)
        if downloader.download_dataset():
            downloader.create_numpy_dataset(save_path=numpy_path)
        else:
            raise RuntimeError("Failed to download test dataset")

    return numpy_path


def test_cpu_processing(data_path: str, batch_size: int = 100, bit_width: int = 4) -> tuple:
    """
    Test CPU processing and timing
    """
    start_time = time.time()

    # Initialize CPU DataManager (CPU processing, CPU storage)
    manager = DataManager(
        data_path=data_path,
        states=bit_width,
        batch_size=batch_size,
        process_device=Device.CPU,
        storage_device=Device.CPU,
    )

    # Load and downsample data
    manager.load_data()
    manager.downsample()

    # Get samples for validation
    samples = manager.get_samples()

    end_time = time.time()
    cpu_time = end_time - start_time

    print(f"CPU processing time: {cpu_time:.4f} seconds")
    print(f"CPU samples shape: {samples.shape}, dtype: {samples.dtype}")

    return samples.array, cpu_time


def test_gpu_processing(data_path: str, batch_size: int = 100, bit_width: int = 4) -> tuple:
    """
    Test GPU processing and timing
    """
    if not cuda.is_available():
        print("CUDA is not available, skipping GPU test")
        return None, 0

    start_time = time.time()
    manager = DataManager(
        data_path=data_path,
        states=bit_width,
        batch_size=batch_size,
        process_device=Device.GPU,
        storage_device=Device.CPU,
    )

    manager.load_data()
    manager.downsample()

    samples = manager.get_samples()

    cuda.synchronize()
    end_time = time.time()
    gpu_time = end_time - start_time

    print(f"GPU processing time: {gpu_time:.4f} seconds")
    print(f"GPU samples shape: {samples.shape}, dtype: {samples.dtype}")

    return samples.array, gpu_time


def compare_results(cpu_result, gpu_result, cpu_time, gpu_time):
    """
    Compare results from CPU and GPU processing
    """
    if gpu_result is None:
        print("GPU test was skipped, cannot compare results")
        return

    if np.array_equal(cpu_result, gpu_result):
        print("✅ CPU and GPU results are identical")
    else:
        print("❌ CPU and GPU results differ")
        diff = np.abs(cpu_result.astype(int) - gpu_result.astype(int))
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean()}")

    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")

    labels = ["CPU", "GPU"]
    times = [cpu_time, gpu_time]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.title('CPU vs GPU Processing Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)

    for i, time_val in enumerate(times):
        plt.text(i, time_val + 0.05, f"{time_val:.4f}s",
                 ha='center', va='bottom', fontweight='bold')

    if gpu_time > 0:
        plt.text(0.5, max(times) * 0.5, f"{speedup:.2f}x speedup",
                 ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("data_manager_benchmark.png")
    plt.show()


if __name__ == "__main__":
    batch_size = 100
    bit_width = 4
    data_path = download_test_data()

    print("\n=== Testing DataManager with CPU and GPU ===")
    cpu_result, cpu_time = test_cpu_processing(data_path, batch_size, bit_width)

    print("\n-------------------------------------------")

    gpu_result, gpu_time = test_gpu_processing(data_path, batch_size, bit_width)

    print("\n=== Results ===")
    compare_results(cpu_result, gpu_result, cpu_time, gpu_time)
