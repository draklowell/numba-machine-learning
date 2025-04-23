import numpy as np
import numba as nb


@nb.njit(nb.uint16[:](nb.uint16, nb.uint16), inline="always")
def _compute_mod_table(axis_size, padding_size):
    mod_table = np.empty((axis_size + 2 * padding_size), dtype=np.uint16)
    for i in range(padding_size):
        mod_table[i] = axis_size - i - 1
    for i in range(axis_size):
        mod_table[i + padding_size] = i
    for i in range(padding_size):
        mod_table[i + axis_size + padding_size] = i

    return mod_table


@nb.njit(
    nb.uint8[:, :, :](nb.uint8[:, :, :], nb.uint8[:, :], nb.uint16, nb.int8[:, :]),
    parallel=True,
)
def apply_cellular_automata(
    images: np.ndarray, rules: np.ndarray, iterations: int, neighborhood: np.ndarray
) -> np.ndarray:
    """
    Core function. Applies a set of rules multiple times to a batch of images using
    a neighborhood-based approach.

    Batch sizes is recommended to be at least 64 for performance reasons.

    Arguments:
        images: A 3D array of shape (batch_size, rows, cols) representing the images.
        rules: A 2D array of shape (num_states, num_states**num_neighbors) representing the rules.
        iterations: The number of iterations to apply the rules.
        neighborhood: A 2D array of shape (num_neighbors, 2) representing the neighborhood offsets.
            Each row contains the (row_offset, col_offset) for a neighbor.

    Returns:
        A 3D array of the same shape as `images` after applying the rules.
    """
    # Double buffer for swapping
    buffer = np.empty_like(images)
    batch_size, rows, cols = images.shape
    states_num = rules.shape[0]

    # Safety padding
    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()

    # Generate mod lookup tables
    mod_row = _compute_mod_table(rows, prow)
    mod_col = _compute_mod_table(cols, pcol)

    # Iterate over each image in the batch (in parallel)
    for idx in nb.prange(batch_size):
        # Initial double buffer setup
        source, target = images[idx], buffer[idx]

        # Iterate over the number of iterations (sequentially)
        for _ in range(iterations):
            # Iterate over each row in the image (in parallel)
            for row in nb.prange(rows):
                # Iterate over each pixel in the row (sequentially)
                for col in range(cols):

                    # Compute transition index based on neighborhood
                    transition = 0
                    shift = 0
                    for nidx in range(len(neighborhood)):
                        nrow, ncol = neighborhood[nidx]
                        # Use mod lookup tables to handle wrapping
                        transition |= (
                            source[
                                mod_row[row + nrow + prow],
                                mod_col[col + ncol + pcol],
                            ]
                            << shift
                        )
                        shift <<= states_num

                    # Apply the rule to determine the new state
                    target[row, col] = rules[source[row, col], transition]

            # Swap buffers for the next iteration
            source, target = target, source

    # Return right buffer from the double buffer setup
    return images if iterations % 2 == 0 else buffer
