import numpy as np
from numba import cuda


@cuda.jit()
def _kernel(source, target):
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

    x_reshaped = x.reshape(-1)

    threads = 1024
    blocks = (x_reshaped.shape[0] + threads - 1) // threads

    y_reshaped = cuda.device_array(x_reshaped.shape, dtype=dtype)
    _kernel[blocks, threads, stream](x_reshaped, y_reshaped)
    return y_reshaped.reshape(x.shape)
