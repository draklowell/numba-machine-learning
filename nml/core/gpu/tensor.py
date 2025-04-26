import numpy as np
from numba import cuda


@cuda.jit()
def cast_kernel(source, target):
    idx = cuda.grid(1)
    if idx >= source.shape[0]:
        return

    target[idx] = source[idx]


def apply_cast_gpu(
    x,
    dtype,
    stream,
):
    dtype = np.dtype(dtype)
    if x.dtype == dtype:
        return x

    # Flatten
    x_reshaped = x.reshape(-1)

    threads = 1024  # CUDA threads per block, hardcoded from documentation
    blocks = (x_reshaped.shape[0] + threads - 1) // threads

    # Flattened target
    y_reshaped = cuda.device_array(x_reshaped.shape, dtype=dtype, stream=stream)

    cast_kernel[blocks, threads, stream](x_reshaped, y_reshaped)

    # Unflatten
    return y_reshaped.reshape(x.shape)
