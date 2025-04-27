import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from data.core.quiantize_cpu import CPUStateDownSampler
from data.core.quiantize_gpu import CUDAStateDownSampler


def test_quantization_accuracy():
    """Test that CPU and GPU quantization produce the same results"""
    print("Testing quantization accuracy...")

    bit_width = 4

    data = np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8)
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


def benchmark_quantization_speed_optimized(sizes, trials=3):
    """Benchmark CPU and GPU quantization speed for different image sizes (optimized)"""
    print("Benchmarking quantization speed (optimized)...")

    bit_width = 4
    cpu_times = []
    gpu_times = []

    cpu_quantizer = CPUStateDownSampler(bit_width)
    gpu_quantizer = CUDAStateDownSampler(bit_width)

    for size in sizes:
        print(f"Testing image size: {size}x{size}")
        d_buf = cuda.device_array((size, size), dtype=np.uint8)
        cpu_trial_times = []
        for _ in range(trials):
            data = np.random.randint(0, 256, size=(size, size), dtype=np.uint8)

            start_time = time.time()
            cpu_quantizer(data.copy())
            end_time = time.time()

            cpu_trial_times.append(end_time - start_time)
        cpu_time = sum(cpu_trial_times) / trials
        cpu_times.append(cpu_time)
        gpu_trial_times = []
        for _ in range(trials):
            data = np.random.randint(0, 256, size=(size, size), dtype=np.uint8)
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
    plt.title("Quantization Speed: CPU vs GPU")
    plt.grid(True)
    plt.legend()
    plt.savefig("quantization_benchmark.png")
    plt.show()


def visualize_quantization(original_image=None):
    """Visualize the effect of quantization on an image"""
    if original_image is None:
        # Create a test gradient image
        size = 512
        x = np.linspace(0, 255, size)
        y = np.linspace(0, 255, size)
        X, Y = np.meshgrid(x, y)
        original_image = X.astype(np.uint8)

    bit_widths = [1, 2, 3, 4]  # You can easily modify this list
    n_images = 1 + len(bit_widths)  # 1 original + quantized images

    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))

    # If only 1 image, axes is not a list
    if n_images == 1:
        axes = [axes]

    # Plot original
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot quantized images
    for ax, bits in zip(axes[1:], bit_widths):
        quantizer = CPUStateDownSampler(bits)
        quantized = quantizer(original_image.copy())

        ax.imshow(quantized, cmap="gray")
        ax.set_title(f"{bits}-bit ({2**bits} states)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("quantization_visual.png")
    plt.show()


if __name__ == "__main__":
    print("Running tests...")

    success = test_quantization_accuracy()
    if not success:
        print("WARNING: CPU and GPU results differ!")

    image_sizes = [28, 128, 256, 512, 1024, 2048]
    sizes, cpu_times, gpu_times = benchmark_quantization_speed_optimized(image_sizes)
    plot_benchmark_results(sizes, cpu_times, gpu_times)

    visualize_quantization()
