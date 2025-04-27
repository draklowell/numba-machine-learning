import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from data.core.quantize_cpu import CPUStateDownSampler
from data.core.quantize_gpu import CUDAStateDownSampler


def test_quantization_accuracy_batch(batch_size=1024):
    """Test that CPU and GPU quantization produce the same results on a batch"""
    print(f"Testing quantization accuracy with batch_size={batch_size}...")

    bit_width = 4
    height, width = 28, 28  # розмір однієї картинки

    data = np.random.randint(0, 256, size=(batch_size, height, width), dtype=np.uint8)
    data_copy = data.copy()

    cpu_quantizer = CPUStateDownSampler(bit_width)
    cpu_result = cpu_quantizer(data.copy())
    gpu_quantizer = CUDAStateDownSampler(bit_width)

    d_data = cuda.to_device(data_copy)
    gpu_quantizer(d_data)
    gpu_result = d_data.copy_to_host()

    matches = np.array_equal(cpu_result, gpu_result)
    print(f"CPU and GPU results match: {matches}")
    if not matches:
        diff = np.abs(cpu_result.astype(int) - gpu_result.astype(int))
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")

    return matches


def benchmark_quantization_speed_batch(sizes, batch_size=1024, trials=3):
    """Benchmark CPU and GPU quantization speed for different image sizes with batch"""
    print(f"Benchmarking quantization speed with batch_size={batch_size}...")

    bit_width = 4
    cpu_times = []
    gpu_times = []

    cpu_quantizer = CPUStateDownSampler(bit_width)
    gpu_quantizer = CUDAStateDownSampler(bit_width)

    for size in sizes:
        print(f"Testing image size: {size}x{size} with batch {batch_size}")
        d_buf = cuda.device_array((batch_size, size, size), dtype=np.uint8)
        cpu_trial_times = []
        for _ in range(trials):
            data = np.random.randint(
                0, 256, size=(batch_size, size, size), dtype=np.uint8
            )

            start_time = time.time()
            cpu_quantizer(data.copy())
            end_time = time.time()

            cpu_trial_times.append(end_time - start_time)

        cpu_time = sum(cpu_trial_times) / trials
        cpu_times.append(cpu_time)

        gpu_trial_times = []
        for _ in range(trials):
            data = np.random.randint(
                0, 256, size=(batch_size, size, size), dtype=np.uint8
            )
            d_buf.copy_to_device(data)

            cuda.synchronize()

            start_time = time.time()
            gpu_quantizer(d_buf)
            cuda.synchronize()
            end_time = time.time()

            gpu_trial_times.append(end_time - start_time)

        gpu_time = sum(gpu_trial_times) / trials
        gpu_times.append(gpu_time)

        print(f"CPU time: {cpu_time:.6f}s, GPU time: {gpu_time:.6f}s")

    return sizes, cpu_times, gpu_times


def plot_benchmark_results(sizes, cpu_times, gpu_times):
    """Plot the benchmark results"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, "b-o", label="CPU")
    plt.plot(sizes, gpu_times, "r-o", label="GPU")
    plt.xlabel("Image Size (pixels per side)")
    plt.ylabel("Time (seconds)")
    plt.title("Quantization Speed: CPU vs GPU (Batch)")
    plt.grid(True)
    plt.legend()
    plt.savefig("quantization_benchmark_batch.png")
    plt.show()


if __name__ == "__main__":
    print("Running batch tests...")

    batch_size = 1024

    success = test_quantization_accuracy_batch(batch_size=batch_size)
    if not success:
        print("WARNING: CPU and GPU results differ!")

    image_sizes = [28, 128, 256, 512]
    sizes, cpu_times, gpu_times = benchmark_quantization_speed_batch(
        image_sizes, batch_size=batch_size
    )
    plot_benchmark_results(sizes, cpu_times, gpu_times)
