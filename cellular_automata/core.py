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
    nb.uint8[:, :, :](
        nb.uint8[:, :, :], # images
        nb.uint8[:],       # rules
        nb.int8[:, :],     # neighborhood
        nb.uint16,         # iterations
        nb.uint8,          # rule_bitwidth
    ),
    parallel=True,
)
def _apply_cellular_automata(
    images: np.ndarray, rules: np.ndarray, neighborhood: np.ndarray, iterations: int, rule_bitwidth: int,
) -> np.ndarray:
    # Double buffer for swapping
    buffer = np.empty_like(images)
    batch_size, rows, cols = images.shape
    num_neighbors = neighborhood.shape[0]

    # Safety padding
    prow = neighborhood[:, 0].max()
    pcol = neighborhood[:, 1].max()

    # Generate mod lookup tables
    mod_row = _compute_mod_table(rows, prow)
    mod_col = _compute_mod_table(cols, pcol)

    # Precompute shifts
    shifts = np.empty((num_neighbors,), dtype=np.uint32)
    for nidx in range(num_neighbors):
        shifts[nidx] = rule_bitwidth * (nidx+1)

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
                    transition = np.uint32(source[row, col])

                    # Compute transition index based on neighborhood
                    for nidx in range(num_neighbors):
                        nrow, ncol = neighborhood[nidx]
                        # Use mod lookup tables to handle wrapping
                        transition |= source[
                            mod_row[row + nrow + prow],
                            mod_col[col + ncol + pcol],
                        ] << shifts[nidx]

                    # Apply the rule to determine the new state
                    target[row, col] = rules[transition]

            # Swap buffers for the next iteration
            source, target = target, source

    # Return right buffer from the double buffer setup
    return images if iterations % 2 == 0 else buffer

def apply_cellular_automata(
    images: np.ndarray,
    rules: np.ndarray,
    neighborhood: np.ndarray,
    iterations: int,
    rule_bitwidth: int,
) -> np.ndarray:
    """
    Core function. Applies a set of rules multiple times to a batch of images using
    a neighborhood-based approach.

    Batch sizes is recommended to be at least 64 for performance reasons.

    Arguments:
        images: A 3D array of shape (batch_size, rows, cols) representing the images.
        rules: A 1D array of size num_states**(num_neighbors+1) representing the rules.
        neighborhood: A 2D array of shape (num_neighbors, 2) representing the neighborhood offsets.
            Each row contains the (row_offset, col_offset) for a neighbor.
        iterations: The number of iterations to apply the rules.
        rule_bitwidth: The bit width of the rules. This determines the number of states.
            num_states is computed as 2**rule_bitwidth.

    Returns:
        A 3D array of the same shape as `images` after applying the rules.
    """
    # Validate input shapes
    if images.ndim != 3:
        raise ValueError("images must be a 3D array (batch_size, rows, cols)")
    if rules.ndim != 1:
        raise ValueError("rules must be a 1D array")
    if neighborhood.ndim != 2 or neighborhood.shape[1] != 2:
        raise ValueError("neighborhood must be a 2D array (num_neighbors, 2)")

    # Validate data types
    if images.dtype != np.uint8:
        raise ValueError("images must be of type np.uint8")
    if rules.dtype != np.uint8:
        raise ValueError("rules must be of type np.uint8")
    if neighborhood.dtype != np.int8:
        raise ValueError("neighborhood must be of type np.int8")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if not isinstance(rule_bitwidth, int) or rule_bitwidth < 1:
        raise ValueError("rule_bitwidth must be a positive integer")

    # Validate rule size
    num_states = 2**rule_bitwidth
    num_neighbors = neighborhood.shape[0]
    if rules.shape[0] != num_states ** (num_neighbors+1):
        raise ValueError(
            "rules size mismatch: expected (2**rule_bitwidth)**(num_neighbors+1)"
        )

    # Apply the cellular automata function
    return _apply_cellular_automata(images, rules, neighborhood, iterations, rule_bitwidth)
