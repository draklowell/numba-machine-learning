from nml.cpu.tensor import CPUTensor
from nml.device import Device
from nml.model import DeferredResults, Model


class Results(DeferredResults):
    """
    Results class for CPU model inference.
    """

    def __init__(self, tensor: CPUTensor):
        self.tensor = tensor

    def wait(self) -> CPUTensor:
        return self.tensor


class CPUModel(Model):
    """
    CPUModel class for CPU-based inference.
    """

    device: Device = Device.CPU

    def _infer(self, tensor: CPUTensor) -> Results:
        ctx = {}
        for unit in self.units:
            tensor = unit(tensor, ctx)

        return Results(tensor)
