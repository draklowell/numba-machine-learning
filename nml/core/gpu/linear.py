from numba import cuda


@cuda.jit()
def _kernel(x, y, weights, biases):
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


def apply_linear_gpu(
    x,
    weights,
    biases,
    stream,
):
    # Transfer weight to GPU
    weights = cuda.to_device(weights, stream=stream)
    biases = cuda.to_device(biases, stream=stream)

    y = cuda.device_array(
        (x.shape[0], weights.shape[1]), dtype=weights.dtype, stream=stream
    )

    # Calculate grid and block sizes (see Numba documentation for details)
    threads_per_block = (32, 32)
    blocks_per_grid_x = (y.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (y.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    _kernel[blocks_per_grid, threads_per_block, stream](x, y, weights, biases)
    return y
