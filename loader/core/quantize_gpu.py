import numpy as np
from numba import cuda, int32, uint8


@cuda.jit
def fused_quantize_batch_kernel(d_images, shift):
    """
    Processes a flattened mini-batch of images.
    Performs a bit-shift for quantization to reduce bit depth.

    Args:
      d_images: 1D device array (dtype uint8) containing the flattened batch.
      shift: int32, number of bits to shift (i.e., 8 - rule_bitwidth).
    """
    idx = cuda.grid(1)
    if idx < d_images.size:
        temp = int32(d_images[idx]) >> shift
        d_images[idx] = uint8(temp)


class CUDAStateDownSampler:
    """
    Initialize a CUDA-based state downsampler.

    Args:
        rule_bitwidth: The number of bits in the output (e.g., 4 for 16 states)
    """

    def __init__(self, rule_bitwidth: int):
        if rule_bitwidth <= 0:
            raise ValueError("rule_bitwidth must be > 0")
        self.states = 1 << rule_bitwidth
        if (self.states & (self.states - 1)) != 0:
            raise ValueError("Only power-of-2 states are supported (1, 2, 4, 8, 16, 32, 64, 128)")
        self.rule_bitwidth = rule_bitwidth
        self.threads_per_block = 128

    def __call__(self, d_array):
        flat = d_array.reshape(-1)
        shift = int32(8 - self.rule_bitwidth)
        blocks = (flat.size + self.threads_per_block - 1) // self.threads_per_block
        fused_quantize_batch_kernel[blocks, self.threads_per_block](flat, shift)
        return d_array


if __name__ == "__main__":
    # Using 4 bits (2^4 = 16 states)
    batch_size = 1024
    height, width = 28, 28
    images = np.random.randint(0, 256, size=(batch_size, height, width), dtype=np.uint8)
    d_images = cuda.to_device(images)
    state_downsampler = CUDAStateDownSampler(4)
    state_downsampler(d_images)
    quantized_images_gpu = d_images.copy_to_host()
    print("Quantized Batch (GPU):")
    print(quantized_images_gpu)
