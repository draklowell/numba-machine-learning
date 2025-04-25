import numba as nb
import numpy as np


@nb.njit(
    nb.uint16[:](nb.uint16, nb.uint16),
    inline="always",
    locals={"mod_table": nb.uint16[:]},
)
def _cpu_ca_compute_mod_table(axis_size, padding_size):
    mod_table = np.empty((axis_size + 2 * padding_size), dtype=np.uint16)
    for i in range(padding_size):
        mod_table[i] = axis_size - i - 1
    for i in range(axis_size):
        mod_table[i + padding_size] = i
    for i in range(padding_size):
        mod_table[i + axis_size + padding_size] = i

    return mod_table


@nb.njit(
    nb.uint8[:, :, :](
        nb.uint8[:, :, :],  # images
        nb.uint8[:],  # rules
        nb.int8[:, :],  # neighborhood
        nb.uint16,  # iterations
        nb.uint8,  # rule_bitwidth
    ),
    parallel=True,
    locals={
        "buffer": nb.uint8[:, :, :],
        "mod_row": nb.uint16[:],
        "mod_col": nb.uint16[:],
        "shifts": nb.uint8[:],
        "source": nb.uint8[:, :],
        "target": nb.uint8[:, :],
        "transition": nb.uint32,
    },
)
def _cpu_ca_apply_cellular_automata(
    images: np.ndarray,
    rules: np.ndarray,
    neighborhood: np.ndarray,
    iterations: int,
    rule_bitwidth: int,
) -> np.ndarray:
    # Double buffer for swapping
    buffer = np.empty_like(images)
    batch_size, rows, cols = images.shape
    num_neighbors = neighborhood.shape[0]

    # Safety padding
    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()

    # Generate mod lookup tables
    mod_row = _cpu_ca_compute_mod_table(rows, prow)
    mod_col = _cpu_ca_compute_mod_table(cols, pcol)

    # Precompute shifts
    shifts = np.empty((num_neighbors,), dtype=np.uint8)
    for nidx in range(num_neighbors):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    # Iterate over each image in the batch (in parallel)
    for idx in nb.prange(batch_size):
        # Initial double buffer setup
        source, target = images[idx], buffer[idx]

        # Iterate over the number of iterations (sequentially)
        for _ in range(iterations):

            # Iterate over each row in the image (in parallel)
            for row in nb.prange(rows):
                # Iterate over each cell in the row (sequentially)
                for col in range(cols):

                    # Create transition index from the current state
                    transition = source[row, col]

                    # Compute transition index based on neighborhood
                    for nidx in range(num_neighbors):
                        nrow, ncol = neighborhood[nidx]
                        # Use mod lookup tables to handle wrapping
                        transition |= (
                            source[
                                mod_row[row + nrow + prow],
                                mod_col[col + ncol + pcol],
                            ]
                            << shifts[nidx]
                        )

                    # Apply the rule to determine the new state
                    target[row, col] = rules[transition]

            # Swap buffers for the next iteration
            source, target = target, source

    # Return right buffer from the double buffer setup
    return images if iterations % 2 == 0 else buffer
