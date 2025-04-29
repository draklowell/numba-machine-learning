from numba import cuda

if not cuda.is_available():
    raise ImportError(
        "CUDA is not available. Please ensure that you have "
        "a compatible GPU and the CUDA toolkit installed."
    )

from nml.gpu.activation import apply_activation
from nml.gpu.cast import apply_cast
from nml.gpu.cellular_automata import apply_cellular_automata
from nml.gpu.linear import apply_linear
from nml.gpu.softmax import apply_softmax
from nml.gpu.tensor import GPUTensor

__all__ = (
    "apply_activation",
    "apply_cast",
    "apply_linear",
    "apply_softmax",
    "apply_cellular_automata",
    "GPUTensor",
)
