import numpy as np
from numpy.typing import NDArray

from nml.core.cpu.cellular_automata import (
    apply_cellular_automata as apply_cellular_automata_cpu,
)
from nml.core.gpu.cellular_automata import (
    apply_cellular_automata as apply_cellular_automata_gpu,
)
from nml.layers.base import InferableLayer, Layer
from nml.parameters import TensorParameter


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
    _neighborhood: NDArray
    _iterations: int | None

    def __init__(
        self,
        name: str,
        rule_bitwidth: int,
        neighborhood: NDArray,
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

    def infer(self, x: NDArray) -> NDArray:
        iterations = self._iterations
        if iterations is None:
            iterations = self._get_parameter("iterations")

        return apply_cellular_automata_cpu(
            x,
            self._get_parameter("rules"),
            self._neighborhood,
            np.uint16(iterations),
            np.uint8(self._rule_bitwidth),
        )

    def infer_cuda(self, x, stream):
        iterations = self._iterations
        if iterations is None:
            iterations = self._get_parameter("iterations")

        return apply_cellular_automata_gpu(
            x,
            self._get_parameter("rules"),
            self._neighborhood,
            np.uint16(iterations),
            np.uint8(self._rule_bitwidth),
            stream=stream,
        )


class CellularAutomata(Layer):
    """
    A layer descriptor for the Cellular Automata layer.
    This class is used to create and configure the Cellular Automata layer.
    """

    name = "cellular_automata"
    _rule_bitwidth: int
    _neighborhood: NDArray
    _iterations: int | None

    def __init__(
        self,
        rule_bitwidth: int = 1,
        neighborhood: str | NDArray = "moore_1",
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
