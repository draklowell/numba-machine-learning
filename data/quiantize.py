import numpy as np
from functools import lru_cache
from numpy.typing import NDArray


@lru_cache(maxsize=None)
def get_lut(states: int) -> NDArray:
    arr = np.arange(256, dtype=np.int32)
    lut = (arr * states) // 256
    return lut.astype(np.uint8)


def quantize_inplace(x: NDArray, states: int) -> None:

    if (states & (states - 1)) == 0:
        shift = 8 - int(np.log2(states))
        x[:] >>=shift
    else:
        lut = get_lut(states)
        x[:] = lut[x]


class StateDownSampler:

    def __init__(self, states: int):
        if states <= 1:
            raise ValueError("The number of states should be > 1")
        self.states = states


    def __call__(self, image: NDArray) -> NDArray:
        data = image.copy() if not image.flags["WRITEABLE"] else image
        quantize_inplace(data, self.states)
        return data

if __name__ == "__main__":
    state = StateDownSampler(10)
    large_array = np.random.randint(0, 256, size=(784), dtype=np.uint8)
    print("Before quantization:", large_array)
    state(large_array)
    print("After quantization:", large_array)