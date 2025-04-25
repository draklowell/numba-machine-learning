import numba as nb
import numpy as np
from numba import cuda
from numpy.typing import NDArray

from nml.core.cpu.cellular_automata import compute_mod_table


def apply_cellular_automata(
    images,
    rules: NDArray,
    neighborhood: NDArray,
    iterations: np.uint16,
    rule_bitwidth: np.uint8,
    stream,
):
    shape = images.shape

    @cuda.jit()
    def _kernel(
        buffer,
        rules,
        neighborhood,
        mod_row,
        mod_col,
        shifts,
        iterations: int,
        prow: int,
        pcol: int,
    ):
        # Get thread indices
        idx = cuda.blockIdx.x
        row = cuda.threadIdx.x
        col = cuda.threadIdx.y
        # Check if indices are within bounds
        if idx >= shape[0] or row >= shape[1] or col >= shape[2]:
            return

        # Create shared buffers for faster access
        buffer_source = cuda.shared.array(shape[1:], dtype=nb.uint8)
        buffer_source[row, col] = buffer[idx, row, col]

        buffer_target = cuda.shared.array(shape[1:], dtype=nb.uint8)

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

    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()
    mod_row = compute_mod_table(shape[1], prow)
    mod_col = compute_mod_table(shape[2], pcol)
    shifts = np.empty((neighborhood.shape[0],), dtype=np.uint8)
    for nidx in range(neighborhood.shape[0]):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    mod_row = cuda.to_device(mod_row)
    mod_col = cuda.to_device(mod_col)
    shifts = cuda.to_device(shifts)
    neighborhood = cuda.to_device(neighborhood)
    rules = cuda.to_device(rules)

    _kernel[images.shape[0], images.shape[1:], stream](
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
