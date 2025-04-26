import numba as nb
import numpy as np
from numba import cuda
from numpy.typing import NDArray

from nml.core.cpu.cellular_automata import compute_mod_table


@cuda.jit()
def _kernel_optimized(
    batches,
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
    bidx = cuda.blockIdx.x
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Check if indices are within bounds
    if bidx >= batches.shape[0] or row >= batches.shape[1] or col >= batches.shape[2]:
        return

    # Create shared buffers for faster access and write from global to shared
    source = cuda.shared.array((32, 32), dtype=nb.uint8)
    source[row, col] = batches[bidx, row, col]

    target = cuda.shared.array((32, 32), dtype=nb.uint8)

    # Synchronize copy from global array to shared one
    cuda.syncthreads()

    for _ in range(iterations):
        # Calculate transition index
        transition = nb.uint32(source[row, col])
        for nidx, (nrow, ncol) in enumerate(neighborhood):
            # Use mod lookup tables to handle wrapping
            transition |= (
                nb.uint32(
                    source[
                        mod_row[row + nrow + prow],
                        mod_col[col + ncol + pcol],
                    ]
                )
                << shifts[nidx]
            )

        # Apply the rule
        target[row, col] = rules[transition]

        # Synchronize each iteration
        cuda.syncthreads()

        # Swap buffers
        source, target = target, source

    # Write back to global memory
    batches[bidx, row, col] = source[row, col]


def apply_cellular_automata_gpu(
    batches,
    rules: NDArray,
    neighborhood: NDArray,
    iterations: np.uint16,
    rule_bitwidth: np.uint8,
    stream,
):
    if batches.shape[1] > 32 or batches.shape[2] > 32:
        raise NotImplementedError(
            "GPU cellular automata only supports images with height and width <= 32"
        )

    # Preperation for kernel launch
    # Precompute shift and mod tables
    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()
    mod_row = compute_mod_table(batches.shape[1], prow)
    mod_col = compute_mod_table(batches.shape[2], pcol)
    shifts = np.empty((neighborhood.shape[0],), dtype=np.uint8)
    for nidx in range(neighborhood.shape[0]):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    # Transfer tables and weights to GPU
    mod_row = cuda.to_device(mod_row, stream=stream)
    mod_col = cuda.to_device(mod_col, stream=stream)
    shifts = cuda.to_device(shifts, stream=stream)
    neighborhood = cuda.to_device(neighborhood, stream=stream)
    rules = cuda.to_device(rules, stream=stream)

    _kernel_optimized[batches.shape[0], batches.shape[1:], stream](
        batches,
        rules,
        neighborhood,
        mod_row,
        mod_col,
        shifts,
        iterations,
        prow,
        pcol,
    )
    return batches
