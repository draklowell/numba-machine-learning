import os
import sys
import time
from pathlib import Path
import urllib.request
import gzip
import shutil

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from numba import cuda

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from data.core.quantize_cpu import CPUStateDownSampler
from data.core.quantize_gpu import CUDAStateDownSampler
from data.extractor.manager import DataManager, StorageDevice
from data.protocols.loader import Loader


class MNISTLoader(Loader):
    """Custom loader for MNIST dataset using python-mnist library"""

    def __init__(self, mnist_path, dataset_type='train'):
        self.mnist_path = mnist_path
        self.dataset_type = dataset_type
        self._data = None
        self._length = 0

    def load(self):
        if self._data is None:
            mndata = MNIST(self.mnist_path)

            if self.dataset_type == 'train':
                images, labels = mndata.load_training()
            else:
                images, labels = mndata.load_testing()

            # Convert to numpy array and reshape to (N, 28, 28)
            self._data = np.array(images, dtype=np.uint8).reshape(-1, 28, 28)
            self._length = len(self._data)

        return self._data

    def __len__(self):
        if self._data is None:
            self.load()
        return self._length


class CPUQuantizeTransform:
    """CPU-based quantization transform"""

    def __init__(self, bit_width=4):
        self.quantizer = CPUStateDownSampler(bit_width)

    def __call__(self, x):
        return self.quantizer(x)


class CUDAQuantizeTransform:
    """CUDA-based quantization transform"""

    def __init__(self, bit_width=4):
        self.quantizer = CUDAStateDownSampler(bit_width)

    def __call__(self, x):
        self.quantizer(x)
        return x


def download_mnist_dataset(mnist_path):
    """
    Download MNIST dataset if it doesn't exist

    Args:
        mnist_path: Path to store MNIST data
    """
    # Original MNIST dataset URLs
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte.gz': 't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte.gz': 't10k-labels-idx1-ubyte'
    }

    # Mirror URLs if the primary URL doesn't work
    backup_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]

    os.makedirs(mnist_path, exist_ok=True)

    # Check if files already exist
    all_files_exist = all(os.path.exists(os.path.join(mnist_path, f)) for f in files.values())
    if all_files_exist:
        print("MNIST dataset already downloaded")
        return

    print("Downloading MNIST digits dataset...")
    for gz_file, output_file in files.items():
        output_path = os.path.join(mnist_path, output_file)
        if not os.path.exists(output_path):
            # Try primary URL first
            url = base_url + gz_file
            gz_path = os.path.join(mnist_path, gz_file)

            downloaded = False
            # Try primary URL
            try:
                print(f"Downloading {gz_file}...")
                urllib.request.urlretrieve(url, gz_path)
                downloaded = True
            except Exception as e:
                print(f"Primary URL failed: {e}")

                # Try backup URLs
                for backup_base_url in backup_urls:
                    if downloaded:
                        break

                    try:
                        backup_url = backup_base_url + gz_file
                        print(f"Trying backup URL: {backup_url}")
                        urllib.request.urlretrieve(backup_url, gz_path)
                        downloaded = True
                    except Exception as backup_e:
                        print(f"Backup URL failed: {backup_e}")

            if not downloaded:
                print("All download attempts failed.")
                print("Please manually download the MNIST dataset and place in:", mnist_path)
                return

            # Extract the file
            print(f"Extracting {gz_file}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the gzip file
            os.remove(gz_path)

    print("MNIST digits dataset download complete")


def benchmark_quantization(mnist_path, bit_width=4, batch_sizes=None):
    """
    Benchmark CPU vs GPU quantization on MNIST data

    Args:
        mnist_path: Path to MNIST data
        bit_width: Quantization bit width
        batch_sizes: List of batch sizes to test
    """
    if batch_sizes is None:
        batch_sizes = [64, 128, 256, 512, 1024]

    # Create loaders
    mnist_loader = MNISTLoader(mnist_path)
    print(f"MNIST dataset size: {len(mnist_loader)}")

    # Results storage
    cpu_times = []
    gpu_times = []

    # Test different batch sizes
    for batch_size in batch_sizes:
        print(f"\nBenchmarking with batch_size={batch_size}")

        # CPU benchmark
        cpu_transform = CPUQuantizeTransform(bit_width)
        cpu_data_manager = DataManager(
            loader=mnist_loader,
            transform=cpu_transform,
            storage_device=StorageDevice.CPU,
            return_device=StorageDevice.CPU,
            prefetch=False
        )

        # Prepare data
        cpu_data_manager.prepare()

        # Run CPU benchmark
        start_time = time.time()
        for _ in range(5):  # Get multiple batches to average the time
            cpu_samples = cpu_data_manager.get_samples(batch_size)
        cpu_time = (time.time() - start_time) / 5
        cpu_times.append(cpu_time)
        print(f"CPU time: {cpu_time:.6f}s")

        # Check if CUDA is available
        if cuda.is_available():
            # GPU benchmark
            cuda_transform = CUDAQuantizeTransform(bit_width)
            gpu_data_manager = DataManager(
                loader=mnist_loader,
                transform=cuda_transform,
                storage_device=StorageDevice.CUDA,
                return_device=StorageDevice.CUDA,
                prefetch=False
            )

            # Prepare data
            gpu_data_manager.prepare()

            # Run GPU benchmark
            start_time = time.time()
            for _ in range(5):
                gpu_samples = gpu_data_manager.get_samples(batch_size)
                cuda.synchronize()
            gpu_time = (time.time() - start_time) / 5
            gpu_times.append(gpu_time)
            print(f"GPU time: {gpu_time:.6f}s")
        else:
            print("CUDA is not available. Skipping GPU benchmark.")
            gpu_times.append(None)

    return batch_sizes, cpu_times, gpu_times


def plot_benchmark_results(batch_sizes, cpu_times, gpu_times):
    """Plot benchmark results"""
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, cpu_times, "b-o", label="CPU")

    # Only plot GPU times if available
    if all(time is not None for time in gpu_times):
        plt.plot(batch_sizes, gpu_times, "r-o", label="GPU")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("MNIST Quantization Speed: CPU vs GPU")
    plt.grid(True)
    plt.legend()
    plt.savefig("mnist_quantization_benchmark.png")
    plt.show()


def benchmark_raw_kernel(mnist_path, bit_width=4, batch_size=10000, iterations=100):
    """Benchmark just the kernel execution without DataManager overhead"""
    # Load data
    mnist_loader = MNISTLoader(mnist_path)
    data = mnist_loader.load()[:batch_size].copy()
    print(f"Running raw kernel benchmark with {batch_size} images ({data.size} pixels)")

    # CPU benchmark
    cpu_quantizer = CPUStateDownSampler(bit_width)
    # Warmup
    cpu_quantizer(data.copy())

    cpu_start = time.time()
    for _ in range(iterations):
        cpu_result = cpu_quantizer(data.copy())
    cpu_time = (time.time() - cpu_start) / iterations

    # GPU benchmark (if available)
    if not cuda.is_available():
        print("CUDA is not available. Skipping GPU benchmark.")
        return cpu_time, None, None

    gpu_quantizer = CUDAStateDownSampler(bit_width)
    # Warmup
    d_data = cuda.to_device(data.copy())
    gpu_quantizer(d_data)
    cuda.synchronize()

    # Measure transfer time
    transfer_start = time.time()
    d_data = cuda.to_device(data.copy())
    cuda.synchronize()
    transfer_time = time.time() - transfer_start

    # Measure kernel time
    kernel_start = time.time()
    for _ in range(iterations):
        gpu_quantizer(d_data)
    cuda.synchronize()
    kernel_time = (time.time() - kernel_start) / iterations

    # Total GPU time
    gpu_time = transfer_time + kernel_time

    print(f"CPU time: {cpu_time:.6f}s")
    print(f"GPU kernel time: {kernel_time:.6f}s")
    print(f"GPU transfer time: {transfer_time:.6f}s")
    print(f"GPU total time: {gpu_time:.6f}s")
    print(f"Speedup (kernel only): {cpu_time/kernel_time:.2f}x")
    print(f"Speedup (with transfer): {cpu_time/gpu_time:.2f}x")

    return cpu_time, kernel_time, transfer_time


def verify_quantization_accuracy(mnist_path, bit_width=4, batch_size=100):
    """
    Verify that CPU and GPU quantization produce the same results
    """
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available. Skipping verification.")
        return False

    print(f"\nVerifying quantization accuracy with batch_size={batch_size}...")

    # Load some data
    mnist_loader = MNISTLoader(mnist_path)
    data = mnist_loader.load()[:batch_size].copy()

    # CPU quantization
    cpu_quantizer = CPUStateDownSampler(bit_width)
    cpu_result = cpu_quantizer(data.copy())

    # GPU quantization
    gpu_quantizer = CUDAStateDownSampler(bit_width)
    d_data = cuda.to_device(data.copy())
    gpu_quantizer(d_data)
    gpu_result = d_data.copy_to_host()

    # Compare results
    matches = np.array_equal(cpu_result, gpu_result)
    print(f"CPU and GPU results match: {matches}")
    if not matches:
        diff = np.abs(cpu_result.astype(int) - gpu_result.astype(int))
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")

    return matches


def benchmark_full_dataset(mnist_path, bit_width=4, iterations=5):
    """Benchmark processing the entire MNIST dataset at once"""
    # Load all data
    mnist_loader = MNISTLoader(mnist_path)
    data = mnist_loader.load().copy()  # Get all 60,000 images
    print(f"Processing full MNIST dataset: {len(data)} images ({data.size} pixels)")

    # CPU benchmark
    cpu_quantizer = CPUStateDownSampler(bit_width)
    cpu_start = time.time()
    for _ in range(iterations):
        cpu_result = cpu_quantizer(data.copy())
    cpu_time = (time.time() - cpu_start) / iterations

    # GPU benchmark
    if not cuda.is_available():
        print("CUDA is not available. Skipping GPU benchmark.")
        return cpu_time, None, None, None

    gpu_quantizer = CUDAStateDownSampler(bit_width)
    # Transfer, process, and retrieve
    transfer_start = time.time()
    d_data = cuda.to_device(data.copy())
    cuda.synchronize()
    transfer_time = time.time() - transfer_start

    kernel_start = time.time()
    gpu_quantizer(d_data)
    cuda.synchronize()
    kernel_time = time.time() - kernel_start

    # Full GPU time including result retrieval
    retrieve_start = time.time()
    gpu_result = d_data.copy_to_host()
    cuda.synchronize()
    retrieve_time = time.time() - retrieve_start

    total_gpu_time = transfer_time + kernel_time + retrieve_time

    print(f"CPU full dataset time: {cpu_time:.6f}s")
    print(f"GPU kernel time: {kernel_time:.6f}s")
    print(f"GPU transfer time: {transfer_time:.6f}s")
    print(f"GPU retrieve time: {retrieve_time:.6f}s")
    print(f"GPU total time: {total_gpu_time:.6f}s")
    print(f"Speedup (kernel only): {cpu_time/kernel_time:.2f}x")
    print(f"Speedup (total): {cpu_time/total_gpu_time:.2f}x")

    return cpu_time, kernel_time, transfer_time, retrieve_time


if __name__ == "__main__":
    # Path to MNIST data folder
    mnist_path = "data/datasets/mnist"

    # Download MNIST dataset if needed
    download_mnist_dataset(mnist_path)

    # First verify that CPU and GPU produce the same results
    print("\n--- Verifying CPU/GPU Result Accuracy ---")
    is_accurate = verify_quantization_accuracy(mnist_path, bit_width=4)
    if not is_accurate:
        print("CPU and GPU results don't match! Benchmark aborted.")
        exit(1)

    # Run raw kernel benchmark with larger batch size
    print("\n--- Running Raw Kernel Benchmark ---")
    cpu_time, gpu_kernel_time, gpu_transfer_time = benchmark_raw_kernel(
        mnist_path, bit_width=4, batch_size=10000, iterations=20
    )
    print("--- Raw Kernel Benchmark Complete ---\n")

    # Run benchmark with full MNIST dataset
    print("\n--- Running Full Dataset Benchmark ---")
    cpu_full, gpu_kernel_full, gpu_transfer_full, gpu_retrieve_full = benchmark_full_dataset(
        mnist_path, bit_width=4, iterations=3
    )
    print("--- Full Dataset Benchmark Complete ---\n")

    # Run the full benchmark with different batch sizes
    print("\n--- Running Batch Size Benchmark Series ---")
    batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch_sizes, cpu_times, gpu_times = benchmark_quantization(
        mnist_path, bit_width=4, batch_sizes=batch_sizes
    )
    print("--- Batch Size Benchmark Complete ---\n")

    # Plot the results
    plot_benchmark_results(batch_sizes, cpu_times, gpu_times)