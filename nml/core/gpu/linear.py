from numba import cuda


@cuda.jit()
def _kernel(x, y, weights, biases):
    # Get thread indices
    idx, row = cuda.grid(2)
    # Check if indices are within bounds
    if idx >= y.shape[0] or row >= y.shape[1]:
        return

    y[idx, row] = biases[row]
    for col in range(weights.shape[0]):
        y[idx, row] += x[idx, col] * weights[col, row]


def apply_linear_gpu(
    x,
    weights,
    biases,
    stream,
):
    weights = cuda.to_device(weights)
    biases = cuda.to_device(biases)

    y = cuda.device_array((x.shape[0], weights.shape[1]), dtype=weights.dtype)

    threads_per_block = (32, 32)
    blocks_per_grid_x = (y.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (y.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    _kernel[blocks_per_grid, threads_per_block, stream](x, y, weights, biases)
    return y
