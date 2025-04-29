import numpy as np

from nml.cpu import CPUTensor
from nml.cpu import apply_cellular_automata as apply_cellular_automata_cpu
from nml.device import Device
from nml.parameter import Parameter
from nml.tensor import Tensor
from nml.units.base import UnitWithWeights

try:
    from nml.gpu import GPUTensor
    from nml.gpu import apply_cellular_automata as apply_cellular_automata_gpu
except ImportError:
    apply_cellular_automata_gpu = None


class CellularAutomataUnit(UnitWithWeights):
    """
    Cellular Automata Unit for applying cellular automata rules to a tensor.
    """

    _rule_bitwidth: int
    _neighborhood: list[tuple[int, int]]
    _iterations: int | None
    _prow: int
    _pcol: int

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        rule_bitwidth: int,
        neighborhood: list[tuple[int, int]],
        iterations: int | None = None,
    ):
        if len(shape) != 2:
            raise ValueError("CellularAutomataUnit only supports 2D input tensors.")

        if dtype != np.dtype("uint8"):
            raise ValueError("CellularAutomataUnit only supports uint8 input tensors.")

        self._rule_bitwidth = rule_bitwidth
        self._iterations = iterations
        self._prow, self._pcol = map(max, zip(*neighborhood))

        if device == Device.CPU:
            self._neighborhood = CPUTensor.create(neighborhood, dtype=np.int8)
        elif device == Device.GPU and apply_cellular_automata_gpu is not None:
            self._neighborhood = GPUTensor.create(neighborhood, dtype=np.int8)
        else:
            raise NotImplementedError(
                f"Device {device} is not supported for CellularAutomataUnit."
            )

        states = 2**rule_bitwidth
        transition_space = states ** (len(neighborhood) + 1)
        parameters = [
            Parameter(
                name="rules",
                shape=(transition_space,),
                dtype=np.uint8,
                low=0,
                high=states,
            ),
        ]
        if iterations is None:
            parameters.append(
                Parameter(
                    name="iterations",
                    shape=(),
                    dtype=np.uint16,
                    low=1,
                )
            )

        super().__init__(parameters, name, shape, dtype, device)

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        iterations = self._iterations
        if iterations is None:
            iterations = self._weights["iterations"].item()

        match self.device:
            case Device.CPU:
                return apply_cellular_automata_cpu(
                    tensor,
                    rules=self._weights["rules"],
                    iterations=iterations,
                    rule_bitwidth=self._rule_bitwidth,
                    neighborhood=self._neighborhood,
                    ctx=ctx,
                )
            case Device.GPU if apply_cellular_automata_gpu is not None:
                return apply_cellular_automata_gpu(
                    tensor,
                    rules=self._weights["rules"],
                    iterations=iterations,
                    prow=self._prow,
                    pcol=self._pcol,
                    rule_bitwidth=self._rule_bitwidth,
                    neighborhood=self._neighborhood,
                    ctx=ctx,
                )

        raise NotImplementedError(
            f"Device {self.device} is not supported for CellularAutomataUnit."
        )
