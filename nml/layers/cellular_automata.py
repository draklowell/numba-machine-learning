import numba as nb
import numpy as np

from nml.layers.base import Layer
from nml.parameters import TensorParameter


@nb.njit(
    nb.uint16[:](nb.uint16, nb.uint16),
    inline="always",
    locals={"mod_table": nb.uint16[:]},
)
def _ca_cpu_compute_mod_table(axis_size, padding_size):
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
def _ca_cpu_apply_cellular_automata(
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
    mod_row = _ca_cpu_compute_mod_table(rows, prow)
    mod_col = _ca_cpu_compute_mod_table(cols, pcol)

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


def _build_neighborhoods():
    neighborhoods = {
        "moore_1": [],
        "moore_2": [],
        "von_neumann_1": [],
        "von_neumann_2": [],
        "cross": [],
    }

    for i in range(-2, 3):
        for j in range(-2, 3):
            if 0 < abs(i) + abs(j) <= 2:
                neighborhoods["von_neumann_2"].append((i, j))
            if 0 < abs(i) + abs(j) <= 1:
                neighborhoods["von_neumann_1"].append((i, j))
                neighborhoods["cross"].append((i, j))

            if abs(i) == 2 and abs(j) == 0:
                neighborhoods["cross"].append((i, j))
            if abs(i) == 0 and abs(j) == 2:
                neighborhoods["cross"].append((i, j))

            if 0 < max(abs(i), abs(j)) <= 2:
                neighborhoods["moore_2"].append((i, j))
            if 0 < max(abs(i), abs(j)) <= 1:
                neighborhoods["moore_1"].append((i, j))

    return {
        name: np.array(coords, dtype=np.int8) for name, coords in neighborhoods.items()
    }


NEIGHBORHOODS = _build_neighborhoods()


class CellularAutomataLayer(Layer):
    """
    A layer that applies a cellular automata transformation to the input tensor.
    This layer uses a set of rules to determine the state of each cell in the
    output tensor based on the states of its neighbors.

    Attributes:
        rule_bitwidth: The number of bits used to represent the state.
        neighborhood: The neighborhood structure used for the cellular automata.
    """

    rule_bitwidth: int
    neighborhood: np.ndarray

    def __init__(
        self,
        name: str = "cellular_automata",
        rule_bitwidth: int = 1,
        neighborhood: str | np.ndarray = "moore_1",
    ):
        if isinstance(neighborhood, str):
            if neighborhood not in NEIGHBORHOODS:
                raise ValueError(
                    f"Invalid neighborhood: {neighborhood}. Available "
                    f"options are: {list(NEIGHBORHOODS.keys())}"
                )

            neighborhood = NEIGHBORHOODS[neighborhood]

        super().__init__(name)

        self.rule_bitwidth = rule_bitwidth
        self.neighborhood = neighborhood

        states = 2**rule_bitwidth
        transition_space = states ** (neighborhood.shape[0] + 1)
        self._create_parameter(
            TensorParameter(
                name="rules",
                shape=(transition_space,),
                dtype=np.uint8,
                low=0,
                high=states,
            )
        )
        self._create_parameter(
            TensorParameter(
                name="iterations",
                shape=(),
                dtype=np.uint16,
                low=1,
            )
        )

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Cellular Automata layer to the input tensor.

        Args:
            x: Input tensor of shape (batch, height, width).

        Returns:
            Output tensor of the same shape as the input.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if x.ndim != 3:
            raise ValueError("Input shape must be 3D (batch, height, width)")
        if not np.can_cast(x.dtype, np.uint8):
            raise ValueError("Input dtype must be castable to uint8")

        return _ca_cpu_apply_cellular_automata(
            x.astype(np.uint8),
            self._get_parameter("rules"),
            self.neighborhood,
            np.uint16(self._get_parameter("iterations")),
            np.uint8(self.rule_bitwidth),
        )
