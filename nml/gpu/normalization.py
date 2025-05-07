from numba import cuda

from nml.gpu.tensor import GPUTensor


@cuda.jit()
def normalization(x, k, b):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = x[idx] * k + b


def apply_normalization(
    tensor: GPUTensor,
    k: float,
    b: float,
    ctx: dict,
):
    # Flatten
    array_reshaped = tensor.array.reshape(-1)

    # Works in place
    threads = 1024  # CUDA threads per block, hardcoded from documentation
    blocks = (array_reshaped.shape[0] + threads - 1) // threads
    normalization[blocks, threads, ctx.get("cuda.stream")](array_reshaped, k, b)

    return tensor
