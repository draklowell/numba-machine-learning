from nml.cpu import CPUTensor
from nml.device import Device
from nml.tensor import Tensor

try:
    from numba import cuda

    from nml.gpu import GPUTensor
except ImportError:
    GPUTensor = None


def copy_to_device(
    source: CPUTensor, device: Device, ctx: dict | None = None
) -> Tensor:
    match device:
        case Device.CPU:
            return CPUTensor(source.array.copy())
        case Device.GPU if GPUTensor is not None:
            if ctx is not None:
                stream = ctx.get("cuda.stream")
            else:
                stream = None

            return GPUTensor(cuda.to_device(source.array, stream=stream))

    raise NotImplementedError(f"Device {device} is not supported")


def copy_to_host(source: Tensor, ctx: dict | None = None) -> CPUTensor:
    match source.device:
        case Device.CPU:
            return CPUTensor(source.array.copy())
        case Device.GPU if GPUTensor is not None:
            if ctx is not None:
                stream = ctx.get("cuda.stream")
            else:
                stream = None

            return CPUTensor(source.array.copy_to_host(stream=stream))

    raise NotImplementedError(f"Device {source.device} is not supported")
