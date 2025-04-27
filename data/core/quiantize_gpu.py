import numpy as np
from numba import cuda, int32, uint8


# --------------------------------------------------------------------------
# Revised Fused CUDA Kernel for Batched Quantization and Clipping
# --------------------------------------------------------------------------
@cuda.jit
def fused_quantize_batch_kernel(d_images, shift, lower_bound, upper_bound):
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

        # Apply clipping fusion:
        if temp < lower_bound:
            temp = lower_bound
        elif temp > upper_bound:
            temp = upper_bound

        # Ensure the result fits in uint8
        d_images[idx] = uint8(temp & 0xFF)


class CUDAStateDownSampler:
    """
    Quantize a batched uint8 device array down to 2**bitwidth states
    using the fused CUDA kernel.
    """

    def __init__(self, bitwidth: int, lower_bound: int = 0, upper_bound: int = 15):
        if not (1 <= bitwidth <= 8):
            raise ValueError("bitwidth must be between 1 and 8")
        self.bitwidth = bitwidth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.threads_per_block = 256

    def __call__(self, d_array):
        flat = d_array.reshape(-1)
        shift = int32(8 - self.bitwidth)
        blocks = (flat.size + self.threads_per_block - 1) // self.threads_per_block
        fused_quantize_batch_kernel[blocks, self.threads_per_block](
            flat, shift, self.lower_bound, self.upper_bound
        )
        return d_array


if __name__ == "__main__":
    # Create a mini-batch of MNIST-like images: 32 images of size 28x28.
    batch_size = 32
    height, width = 28, 28
    images = np.random.randint(0, 256, size=(batch_size, height, width), dtype=np.uint8)

    # Transfer the batch to the GPU.
    d_images = cuda.to_device(images)

    # Quantize on GPU using 4-bit quantization (targeting 16 states) with additional clipping.
    state_downsampler = CUDAStateDownSampler(4, lower_bound=0, upper_bound=15)
    state_downsampler(d_images)

    # Copy the quantized data back to host and print a snippet.
    quantized_images_gpu = d_images.copy_to_host()
    print("Quantized Batch (GPU):")
    print(quantized_images_gpu)
