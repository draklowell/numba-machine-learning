import numpy as np

from nml.cpu import CPUTensor
from nml.device import Device
from nml.parameter import Parameter
from nml.tensor import Tensor
from nml.units.base import UnitWithWeights

try:
    from nml.gpu import apply_linear
except ImportError:
    apply_linear = None


class LinearUnit(UnitWithWeights):
    """
    Linear Unit for applying linear transformation to a tensor.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        output_size: int,
        include_bias: bool = True,
    ):
        if len(shape) != 1:
            raise ValueError("LinearUnit only supports 1D input tensors.")

        parameters = [
            Parameter(
                name="weights",
                shape=(shape[0], output_size),
                dtype=dtype,
            ),
        ]
        if include_bias:
            parameters.append(
                Parameter(
                    name="biases",
                    shape=(output_size,),
                    dtype=dtype,
                )
            )

        super().__init__(parameters, name, (output_size,), dtype, device)

    def infer(self, tensor: Tensor, ctx: dict) -> Tensor:
        match self.device:
            case Device.CPU:
                if "biases" in self._weights:
                    return CPUTensor(
                        tensor.array @ self._weights["weights"].array
                        + self._weights["biases"].array
                    )

                return CPUTensor(tensor.array @ self._weights["weights"].array)
            case Device.GPU if apply_linear is not None:
                return apply_linear(
                    tensor,
                    self._weights["weights"],
                    self._weights.get("biases"),
                    ctx=ctx,
                )

        raise NotImplementedError(
            f"Device {self.device} is not supported for CellularAutomataUnit."
        )
