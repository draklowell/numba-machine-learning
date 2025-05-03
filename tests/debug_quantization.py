import numpy as np
import sys
from pathlib import Path
from numba import cuda

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from loader.core.quantize_cpu import CPUStateDownSampler, quantize_inplace_cpu
from loader.core.quantize_gpu import CUDAStateDownSampler

def debug_quantization_implementations():
    """
    Debug the CPU and GPU quantization implementations to identify differences.
    """
    # Create a simple test array with values ranging from 0-255
    test_size = 100
    test_array = np.arange(0, 256, 256//test_size, dtype=np.uint8)[:test_size]
    test_array = np.tile(test_array, (10, 1)).reshape(10, 10, 10)  # Create a 3D array

    print(f"Test array shape: {test_array.shape}, dtype: {test_array.dtype}")
    print(f"Test array range: {test_array.min()}-{test_array.max()}")

    # Test different bit widths
    for bit_width in [1, 2, 3, 4, 5]:
        print(f"\n--- Testing bit_width={bit_width} ---")
        states = 1 << bit_width
        shift = 8 - bit_width

        # Expected shift calculation
        expected_shift = 8 - int(np.log2(states))
        print(f"Expected shift: {expected_shift}")

        # CPU implementation tests
        cpu_array = test_array.copy()
        cpu_sampler = CPUStateDownSampler(bit_width)
        cpu_result = cpu_sampler(cpu_array)

        # Debug CPU calculation
        calc_shift = 8 - int(np.log2(cpu_sampler.states))
        print(f"CPU calculated shift: {calc_shift}")
        print(f"CPU states: {cpu_sampler.states}")

        # Direct shift test
        direct_array = test_array.copy()
        direct_array = direct_array >> expected_shift

        # GPU implementation test
        gpu_array = test_array.copy()
        d_array = cuda.to_device(gpu_array)
        gpu_sampler = CUDAStateDownSampler(bit_width)
        gpu_sampler(d_array)
        gpu_result = d_array.copy_to_host()

        # Debug GPU calculation
        calc_shift_gpu = 8 - gpu_sampler.rule_bitwidth
        print(f"GPU calculated shift: {calc_shift_gpu}")
        print(f"GPU states: {gpu_sampler.states}")

        # Compare results
        cpu_gpu_equal = np.array_equal(cpu_result, gpu_result)
        cpu_direct_equal = np.array_equal(cpu_result, direct_array)

        print(f"CPU and GPU results match: {cpu_gpu_equal}")
        print(f"CPU and direct shift match: {cpu_direct_equal}")

        if not cpu_gpu_equal:
            # Analyze differences
            diff = np.abs(cpu_result.astype(int) - gpu_result.astype(int))
            print(f"Max difference: {diff.max()}")
            print(f"Mean difference: {diff.mean()}")

            # Sample some differences
            diff_indices = np.where(diff > 0)
            if len(diff_indices[0]) > 0:
                sample_count = min(5, len(diff_indices[0]))
                for i in range(sample_count):
                    idx = tuple(arr[i] for arr in diff_indices)
                    print(f"Sample diff at {idx}: CPU={cpu_result[idx]}, GPU={gpu_result[idx]}, Original={test_array[idx]}")
                    # Show binary representation
                    print(f"  Original: {bin(test_array[idx])}")
                    print(f"  CPU: {bin(cpu_result[idx])}")
                    print(f"  GPU: {bin(gpu_result[idx])}")
                    print(f"  Manual shift ({expected_shift}): {bin(test_array[idx] >> expected_shift)}")

if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available, cannot run test")
        sys.exit(1)

    print("Starting quantization debug test...")
    debug_quantization_implementations()
