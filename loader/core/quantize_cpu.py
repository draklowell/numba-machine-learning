from functools import lru_cache

import numpy as np


def quantize_inplace_cpu(x: np.ndarray, states: int) -> None:
    shift = 8 - int(np.log2(states))
    x[:] >>= shift


class CPUStateDownSampler:
    """
    Initialize a CPU-based state downsampler.
    Args:
        rule_bitwidth: The number of bits in the output (e.g., 4 for 16 states)
    """

    def __init__(self, rule_bitwidth: int):
        if rule_bitwidth <= 0:
            raise ValueError("rule_bitwidth must be > 0")
        self.states = 1 << rule_bitwidth
        if (self.states & (self.states - 1)) != 0:
            raise ValueError("Only power-of-2 states are supported (1, 2, 4, 8, 16, 32, 64, 128)")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        data = image.copy() if not image.flags["WRITEABLE"] else image
        quantize_inplace_cpu(data, self.states)
        return data


if __name__ == "__main__":
    # Using 4 bits (2^4 = 16 states)
    state = CPUStateDownSampler(3)
    large_array = np.random.randint(0, 256, size=(784), dtype=np.uint8)
    print("Before quantization:", large_array)
    state(large_array)
    print("After quantization:", large_array)
