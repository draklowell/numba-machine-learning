from numba import cuda

from nml.gpu.tensor import GPUTensor


@cuda.jit()
def linear(x, y, weights, biases):
    # Get positions
    bidx, row = cuda.grid(2)

    # Check if indices are within bounds
    if bidx >= y.shape[0] or row >= y.shape[1]:
        return

    # Vector by matrix multiplication
    sum_ = biases[row]

    for col in range(weights.shape[0]):
        sum_ += x[bidx, col] * weights[col, row]

    y[bidx, row] = sum_


@cuda.jit()
def linear_no_biases(x, y, weights):
    # Get positions
    bidx, row = cuda.grid(2)

    # Check if indices are within bounds
    if bidx >= y.shape[0] or row >= y.shape[1]:
        return

    # Vector by matrix multiplication
    sum_ = 0

    for col in range(weights.shape[0]):
        sum_ += x[bidx, col] * weights[col, row]

    y[bidx, row] = sum_


def apply_linear(
    tensor: GPUTensor,
    weights: GPUTensor,
    biases: GPUTensor | None,
    ctx: dict,
):
    result = GPUTensor.empty(
        (tensor.shape[0], weights.shape[1]),
        dtype=weights.dtype,
        ctx=ctx,
    )

    # Calculate grid and block sizes (see Numba documentation for details)
    threads_per_block = (32, 32)
    blocks_per_grid_x = (
        result.shape[0] + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blocks_per_grid_y = (
        result.shape[1] + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    if biases is None:
        linear_no_biases[blocks_per_grid, threads_per_block, ctx.get("cuda.stream")](
            tensor.array, result.array, weights.array
        )
    else:
        linear[blocks_per_grid, threads_per_block, ctx.get("cuda.stream")](
            tensor.array, result.array, weights.array, biases.array
        )

    return result
