from numba import cuda

from nml.device import Device
from nml.gpu import GPUTensor
from nml.model import DeferredResults, Model

if not cuda.is_available():
    raise ImportError("CUDA is not available")


class CUDAStream(DeferredResults):
    """
    A class representing a CUDA stream for deferred results.
    """

    def __init__(self, stream, result):
        self.stream = stream
        self.result = result

    def wait(self) -> GPUTensor:
        self.stream.synchronize()
        return self.result.copy_to_host(stream=self.stream)


class GPUModel(Model):
    """
    GPU model for building a neural network.
    """

    device: Device = Device.GPU

    def _infer(self, tensor: GPUTensor) -> CUDAStream:
        stream = cuda.stream()
        ctx = {
            "cuda.stream": stream,
        }
        for unit in self.units:
            tensor = unit(tensor, ctx)

        return CUDAStream(stream, tensor)
