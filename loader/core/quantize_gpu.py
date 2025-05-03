import numpy as np
from numba import cuda, int32, uint8


@cuda.jit
def fused_quantize_batch_kernel(d_images, shift):
    """
    Processes a flattened mini-batch of images.
    Performs:
      1) A bit-shift for quantization, and
      2) A fused clipping operation.

    Args:
      d_images: 1D device array (dtype uint8) containing the flattened batch.
      shift: int32, number of bits to shift (i.e., 8 - bitwidth).
      lower_bound, upper_bound: int32, clipping bounds applied after shifting.
    """
    idx = cuda.grid(1)
    if idx < d_images.size:
        temp = int32(d_images[idx]) >> shift
        d_images[idx] = uint8(temp)


class CUDAStateDownSampler:
    """
    Quantize a batched uint8 device array down to 2**bitwidth states
    using the fused CUDA kernel.
    """

    def __init__(self, bitwidth: int):
        if not (1 <= bitwidth <= 8):
            raise ValueError("bitwidth must be between 1 and 8")
        self.bitwidth = bitwidth
        self.threads_per_block = 128

    def __call__(self, d_array):
        flat = d_array.reshape(-1)
        shift = int32(8 - self.bitwidth)
        blocks = (flat.size + self.threads_per_block - 1) // self.threads_per_block
        fused_quantize_batch_kernel[blocks, self.threads_per_block](flat, shift)
        return d_array


if __name__ == "__main__":
    batch_size = 1024
    height, width = 28, 28
    images = np.random.randint(0, 256, size=(batch_size, height, width), dtype=np.uint8)
    d_images = cuda.to_device(images)
    state_downsampler = CUDAStateDownSampler(4)
    state_downsampler(d_images)
    quantized_images_gpu = d_images.copy_to_host()
    print("Quantized Batch (GPU):")
    print(quantized_images_gpu)
