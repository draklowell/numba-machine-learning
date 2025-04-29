from nml.cpu import CPUTensor
from nml.device import Device
from nml.layers import (
    Cast,
    CellularAutomata,
    Flatten,
    Layer,
    Linear,
    ReLU,
    Reshape,
    Sigmoid,
    Softmax,
    Tanh,
)
from nml.model import DeferredResults, Model
from nml.sequential import Input, Sequential
from nml.tensor import Tensor

try:
    from nml.gpu import GPUTensor
except ImportError:
    GPUTensor = None

__all__ = (
    "CPUTensor",
    "Device",
    "Cast",
    "CellularAutomata",
    "Flatten",
    "Layer",
    "Linear",
    "ReLU",
    "Reshape",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "DeferredResults",
    "Model",
    "Input",
    "Sequential",
    "Tensor",
)

if GPUTensor is not None:
    __all__ += ("GPUTensor",)
