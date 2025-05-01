import numpy as np

from nml.device import Device
from nml.layers.base import Layer
from nml.units import CellularAutomataUnit


def _build_neighborhoods() -> dict[str, list[tuple[int, int]]]:
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

    return neighborhoods


NEIGHBORHOODS = _build_neighborhoods()


class CellularAutomata(Layer):
    """
    A layer descriptor for the Cellular Automata layer.
    This class is used to create and configure the Cellular Automata layer.
    """

    name: str = "cellular_automata"
    _rule_bitwidth: int
    _neighborhood: list[tuple[int, int]]
    _iterations: int | None

    def __init__(
        self,
        rule_bitwidth: int = 1,
        neighborhood: str | list[tuple[int, int]] = "moore_1",
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

    def __call__(
        self, shape: tuple[int, ...], dtype: np.dtype, name: str, device: Device
    ) -> CellularAutomataUnit:
        return CellularAutomataUnit(
            name,
            shape,
            dtype,
            device,
            self._rule_bitwidth,
            self._neighborhood,
            self._iterations,
        )
