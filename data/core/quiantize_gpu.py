import numpy as np
from numba import cuda, uint8, uint32


@cuda.jit
def _quantize_kernel_flat(x, shift):
    """
    A 1-D kernel over a flattened view of the array.
    x     : 1-D device array of uint8
    shift : int32 number of bits to drop (8 - bitwidth)
    """
    idx = cuda.grid(1)
    if idx < x.size:
        x[idx] = (x[idx] >> shift) & 0xFF


class CUDAStateDownSampler:
    """
    Quantize an arbitrary-shaped uint8 device array
    down to 2**bitwidth states
    """

    def __init__(self, bitwidth: int):
        if not (1 <= bitwidth <= 8):
            raise ValueError("bitwidth must be between 1 and 8")
        self.shift = np.int32(8 - bitwidth)
        self.threads_per_block = 1024

    def __call__(self, d_array):
        flat = d_array.reshape((d_array.size,))
        blocks = (flat.size + self.threads_per_block - 1) // self.threads_per_block
        _quantize_kernel_flat[blocks, self.threads_per_block](flat, self.shift)
        return d_array


if __name__ == "__main__":
    # Using 4 bits (2^4 = 16 states)
    state = CUDAStateDownSampler(4)
    large_array = np.random.randint(0, 256, size=(784), dtype=np.uint8)
    d_large_array = cuda.to_device(large_array)
    print("Before quantization:", large_array)
    state(d_large_array)
    print("After quantization:", d_large_array.copy_to_host())
