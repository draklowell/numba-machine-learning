import numba as nb
import numpy as np
from numpy.typing import NDArray

from nml.cpu.tensor import CPUTensor


@nb.njit(
    nb.uint16[:](nb.uint16, nb.uint16),
    inline="always",
    locals={"mod_table": nb.uint16[:]},
)
def compute_mod_table(axis_size, padding_size):
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
def cellular_automata(
    tensor: NDArray,
    rules: NDArray,
    neighborhood: NDArray,
    iterations: int,
    rule_bitwidth: int,
) -> NDArray:
    # Double buffer for swapping
    buffer = np.empty_like(tensor)
    batch_size, rows, cols = tensor.shape
    num_neighbors = neighborhood.shape[0]

    # Safety padding
    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()

    # Generate mod lookup tables
    mod_row = compute_mod_table(rows, prow)
    mod_col = compute_mod_table(cols, pcol)

    # Precompute shifts
    shifts = np.empty((num_neighbors,), dtype=np.uint8)
    for nidx in range(num_neighbors):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    # Iterate over each image in the batch (in parallel)
    for bidx in nb.prange(batch_size):
        # Initial double buffer setup
        source, target = tensor[bidx], buffer[bidx]

        # Iterate over the number of iterations (sequentially)
        for _ in range(iterations):

            # Iterate over each row in the image (in parallel)
            for row in nb.prange(rows):
                # Iterate over each cell in the row (sequentially)
                for col in range(cols):

                    # Create transition index from the current state
                    transition = nb.uint32(source[row, col])

                    # Compute transition index based on neighborhood
                    for nidx in range(num_neighbors):
                        nrow, ncol = neighborhood[nidx]
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

                    # Apply the rule to determine the new state
                    target[row, col] = rules[transition]

            # Swap buffers for the next iteration
            source, target = target, source

    # Return right buffer from the double buffer setup
    return tensor if iterations % 2 == 0 else buffer


def apply_cellular_automata(
    tensor: CPUTensor,
    rules: CPUTensor,
    neighborhood: CPUTensor,
    iterations: int,
    rule_bitwidth: int,
    ctx: dict,
):
    return CPUTensor(
        cellular_automata(
            tensor.array,
            rules.array,
            neighborhood.array,
            np.uint16(iterations),
            np.uint8(rule_bitwidth),
        )
    )
