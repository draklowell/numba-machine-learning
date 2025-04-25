import numba as nb
import numpy as np

from nml.layers.base import InferableLayer, Layer
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


class InferableCellularAutomata(InferableLayer):
    """
    Inferable layer for the Cellular Automata.
    This class is used to apply the Cellular Automata rules to the input tensor.
    """

    _rule_bitwidth: int
    _neighborhood: np.ndarray
    _iterations: int | None

    def __init__(
        self,
        name: str,
        rule_bitwidth: int,
        neighborhood: np.ndarray,
        iterations: int | None = None,
    ):
        self._rule_bitwidth = rule_bitwidth
        self._neighborhood = neighborhood
        self._iterations = iterations

        states = 2**rule_bitwidth
        transition_space = states ** (neighborhood.shape[0] + 1)
        parameters = [
            TensorParameter(
                name="rules",
                shape=(transition_space,),
                dtype=np.uint8,
                low=0,
                high=states,
            ),
        ]
        if iterations is None:
            parameters.append(
                TensorParameter(
                    name="iterations",
                    shape=(),
                    dtype=np.uint16,
                    low=1,
                )
            )
        super().__init__(name, parameters)

    def infer(self, x: np.ndarray) -> np.ndarray:
        iterations = self._iterations
        if iterations is None:
            iterations = self._get_parameter("iterations")

        return _ca_cpu_apply_cellular_automata(
            x,
            self._get_parameter("rules"),
            self._neighborhood,
            np.uint16(iterations),
            np.uint8(self._rule_bitwidth),
        )


class CellularAutomata(Layer):
    """
    A layer descriptor for the Cellular Automata layer.
    This class is used to create and configure the Cellular Automata layer.
    """

    name = "cellular_automata"
    _rule_bitwidth: int
    _neighborhood: np.ndarray
    _iterations: int | None

    def __init__(
        self,
        rule_bitwidth: int = 1,
        neighborhood: str | np.ndarray = "moore_1",
        iterations: int | None = None,
    ):
        if isinstance(neighborhood, str):
            if neighborhood not in NEIGHBORHOODS:
                raise ValueError(
                    f"Invalid neighborhood: {neighborhood}. Available "
                    f"options are: {list(NEIGHBORHOODS.keys())}"
                )

            neighborhood = NEIGHBORHOODS[neighborhood]

        super().__init__()

        self._rule_bitwidth = rule_bitwidth
        self._neighborhood = neighborhood
        self._iterations = iterations

    def build(
        self, idx: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[InferableCellularAutomata, tuple[int, ...], np.dtype]:
        if len(shape) != 2:
            raise ValueError("Input shape must be 2D (height, width)")

        if np.dtype(dtype) != np.dtype("uint8"):
            raise TypeError(f"Input dtype must be uint8, but got: {dtype}")

        return (
            InferableCellularAutomata(
                f"{self.name}_{idx}",
                self._rule_bitwidth,
                self._neighborhood,
                self._iterations,
            ),
            shape,
            dtype,
        )
