import numba
import numpy as np
from numba import cuda
from numpy.typing import NDArray


@cuda.jit()
def quantize_kernel(x, shift) -> None:
    idx = cuda.gridI(1)
    if idx < x.size:
        x[idx] >>= shift


class CUDAtateDownSampler:
    def __init__(self, rule_bitwidth: int):
        if rule_bitwidth <= 1:
            raise ValueError("The number of states should be > 1")
        self.shift = 8 - rule_bitwidth

    def __call__(self, d_array) -> cuda.cudadrv.devicearray.DeviceNDArray:
        threads_per_block = 1024
        blocks_per_grid = (d_array.size + threads_per_block - 1) // threads_per_block
        quantize_kernel[blocks_per_grid, threads_per_block](d_array, self.shift)
        return d_array


if __name__ == "__main__":
    state = CUDAtateDownSampler(10)
    large_array = np.random.randint(0, 256, size=(784), dtype=np.uint8)
    d_large_array = cuda.to_device(large_array)
    print("Before quantization:", large_array)
    state(d_large_array)
    print("After quantization:", d_large_array.copy_to_host())
