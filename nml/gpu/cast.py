import numpy as np
from numba import cuda

from nml.gpu.tensor import GPUTensor


@cuda.jit()
def cast(source, target):
    idx = cuda.grid(1)
    if idx >= source.shape[0]:
        return

    target[idx] = source[idx]


def apply_cast(
    tensor: GPUTensor,
    dtype: np.dtype,
    ctx: dict,
):
    dtype = np.dtype(dtype)
    if tensor.dtype == dtype:
        return tensor

    stream = ctx.get("cuda.stream")

    # Flatten
    array_reshaped = tensor.array.reshape(-1)

    # Result + flatten
    result = GPUTensor.empty(tensor.shape, dtype=dtype, ctx=ctx)
    result_array_reshaped = result.array.reshape(-1)

    threads = 1024  # CUDA threads per block, hardcoded from documentation
    blocks = (array_reshaped.shape[0] + threads - 1) // threads

    cast[blocks, threads, stream](array_reshaped, result_array_reshaped)

    return result
