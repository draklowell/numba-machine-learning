import os
import time
import sys
from pathlib import Path

import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from data.manager.downloader import Downloader
from data.manager.data_manager import DataManager, Device


def benchmark_mnist_quantization():
    """Benchmark CPU vs GPU quantization speed for the entire MNIST dataset"""

    # Set paths
    mnist_path = "data/datasets/mnist"
    train_images_path = os.path.join(mnist_path, "train_images.npy")

    # Download and prepare MNIST dataset if needed
    if not os.path.exists(train_images_path):
        print("Downloading and preparing MNIST dataset...")
        os.makedirs(mnist_path, exist_ok=True)
        downloader = Downloader(mnist_path)
        if downloader.download_dataset():
            downloader.create_numpy_dataset(save_path=train_images_path)
        else:
            print("Failed to download MNIST dataset")
            return

    # Load dataset info to get total size
    train_images = np.load(train_images_path)
    batch_size = train_images.shape[0]  # Use the entire dataset
    print(f"MNIST dataset loaded: {train_images.shape}, using batch size: {batch_size}")

    # Parameters for benchmarking
    bit_width = 4  # 4-bit quantization (16 states)
    trials = 3  # Run multiple trials for more reliable timing

    # Results storage
    cpu_times = []
    gpu_times = []

    # Run trials
    for trial in range(trials):
        print(f"\nTrial {trial+1}/{trials}")

        # Benchmark CPU processing
        print("Benchmarking CPU processing (CPU → CPU)...")
        cpu_manager = DataManager(
            data_path=train_images_path,
            states=bit_width,
            batch_size=batch_size,
            process_device=Device.CPU,
            storage_device=Device.CPU
        )

        # Measure CPU processing time
        start_time = time.time()
        cpu_manager.load_data()
        cpu_manager.downsample()
        cpu_time = time.time() - start_time
        print(f"  CPU processing time: {cpu_time:.4f} seconds")
        cpu_times.append(cpu_time)

        # Benchmark GPU processing
        print("Benchmarking GPU processing (GPU → GPU)...")
        gpu_manager = DataManager(
            data_path=train_images_path,
            states=bit_width,
            batch_size=batch_size,
            process_device=Device.GPU,
            storage_device=Device.GPU
        )

        # Measure GPU processing time (including data transfers)
        start_time = time.time()
        gpu_manager.load_data()
        gpu_manager.downsample()
        cuda.synchronize()  # Ensure all GPU operations are complete
        gpu_time = time.time() - start_time
        print(f"  GPU processing time: {gpu_time:.4f} seconds")
        gpu_times.append(gpu_time)

    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_gpu_time = sum(gpu_times) / len(gpu_times)

    # Print results
    print("\nBenchmark Results:")
    print(f"Average CPU processing time: {avg_cpu_time:.4f} seconds")
    print(f"Average GPU processing time: {avg_gpu_time:.4f} seconds")
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
    print(f"GPU speedup: {speedup:.2f}x faster than CPU")

    # Visualize the results
    plot_results(cpu_times, gpu_times)


def plot_results(cpu_times, gpu_times):
    """Plot benchmark results as a bar chart"""
    avg_cpu = sum(cpu_times) / len(cpu_times)
    avg_gpu = sum(gpu_times) / len(gpu_times)

    labels = ['CPU → CPU', 'GPU → GPU']
    times = [avg_cpu, avg_gpu]
    colors = ['blue', 'orange']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=colors)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}s", ha='center', va='bottom')

    plt.title('MNIST Quantization Performance: CPU vs GPU')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add speedup annotation
    speedup = avg_cpu / avg_gpu if avg_gpu > 0 else float('inf')
    plt.annotate(f'{speedup:.2f}x speedup',
                xy=(1, avg_gpu),
                xytext=(1, (avg_cpu + avg_gpu)/2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center')

    plt.tight_layout()
    plt.savefig('mnist_quantization_benchmark.png')
    plt.show()


if __name__ == "__main__":
    benchmark_mnist_quantization()
