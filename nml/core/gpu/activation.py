import math

from numba import cuda


@cuda.jit()
def tanh_kernel(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = math.tanh(x[idx])


@cuda.jit()
def leaky_relu_kernel(x, alpha):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = x[idx] if x[idx] > 0 else alpha * x[idx]


@cuda.jit()
def relu_kernel(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = max(0, x[idx])


@cuda.jit()
def sigmoid_kernel(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = 1 / (1 + math.exp(-x[idx]))


def apply_activation_gpu(
    kernel,
    x,
    *args,
    stream,
):
    # Flatten
    x_reshaped = x.reshape(-1)

    threads = 1024  # CUDA threads per block, hardcoded from documentation
    blocks = (x_reshaped.shape[0] + threads - 1) // threads
    kernel[blocks, threads, stream](x_reshaped, *args)

    # Unflatten
    return x_reshaped.reshape(x.shape)
