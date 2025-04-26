import numba as nb
import numpy as np
from numba import cuda
from numpy.typing import NDArray

from nml.core.cpu.cellular_automata import compute_mod_table


@cuda.jit()
def _kernel_optimized(
    buffer,
    rules,
    neighborhood,
    mod_row,
    mod_col,
    shifts,
    iterations,
    prow,
    pcol,
):
    # Get thread indices
    idx = cuda.blockIdx.x
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y
    # Check if indices are within bounds
    if idx >= buffer.shape[0] or row >= buffer.shape[1] or col >= buffer.shape[2]:
        return

    # Create shared buffers for faster access
    buffer_source = cuda.shared.array((32, 32), dtype=nb.uint8)
    buffer_source[row, col] = buffer[idx, row, col]

    buffer_target = cuda.shared.array((32, 32), dtype=nb.uint8)

    # Synchronize copy from global array to shared one
    cuda.syncthreads()

    for _ in range(iterations):
        # Calculate transition index
        transition = nb.uint32(buffer_source[row, col])
        for nidx, (nrow, ncol) in enumerate(neighborhood):
            # Use mod lookup tables to handle wrapping
            transition |= (
                nb.uint32(
                    buffer_source[
                        mod_row[row + nrow + prow],
                        mod_col[col + ncol + pcol],
                    ]
                )
                << shifts[nidx]
            )

        buffer_target[row, col] = rules[transition]
        # Synchronize each iteration
        cuda.syncthreads()

        # Swap buffers
        buffer_source, buffer_target = buffer_target, buffer_source

    buffer[idx, row, col] = buffer_source[row, col]


def apply_cellular_automata_gpu(
    images,
    rules: NDArray,
    neighborhood: NDArray,
    iterations: np.uint16,
    rule_bitwidth: np.uint8,
    stream,
):
    if images.shape[1] > 32 or images.shape[2] > 32:
        raise NotImplementedError(
            "GPU cellular automata only supports images with height and width <= 32"
        )

    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()
    mod_row = compute_mod_table(images.shape[1], prow)
    mod_col = compute_mod_table(images.shape[2], pcol)
    shifts = np.empty((neighborhood.shape[0],), dtype=np.uint8)
    for nidx in range(neighborhood.shape[0]):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    mod_row = cuda.to_device(mod_row)
    mod_col = cuda.to_device(mod_col)
    shifts = cuda.to_device(shifts)
    neighborhood = cuda.to_device(neighborhood)
    rules = cuda.to_device(rules)

    _kernel_optimized[images.shape[0], images.shape[1:], stream](
        images,
        rules,
        neighborhood,
        mod_row,
        mod_col,
        shifts,
        iterations,
        prow,
        pcol,
    )
    return images
