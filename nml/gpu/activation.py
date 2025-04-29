import math

from numba import cuda

from nml.gpu.tensor import GPUTensor


@cuda.jit()
def tanh(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = math.tanh(x[idx])


@cuda.jit()
def leaky_relu(x, alpha):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = x[idx] if x[idx] > 0 else alpha * x[idx]


@cuda.jit()
def relu(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = max(0, x[idx])


@cuda.jit()
def sigmoid(x):
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    x[idx] = 1 / (1 + math.exp(-x[idx]))


kernels = {
    "tanh": tanh,
    "leaky_relu": leaky_relu,
    "relu": relu,
    "sigmoid": sigmoid,
}


def apply_activation(
    activation: str,
    tensor: GPUTensor,
    *args,
    ctx: dict,
):
    # Flatten
    array_reshaped = tensor.array.reshape(-1)

    # Works in place
    threads = 1024  # CUDA threads per block, hardcoded from documentation
    blocks = (array_reshaped.shape[0] + threads - 1) // threads
    kernels[activation][blocks, threads, ctx.get("cuda.stream")](array_reshaped, *args)

    return tensor
