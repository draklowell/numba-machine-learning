import random

import numpy as np
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float32,
    xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float32,
)

from nml import GPUTensor


@cuda.jit()
def gaussian_mutation_f64(
    tensor,
    states,
    low: np.number,
    high: np.number,
    rate: np.number,
    strength: np.number,
):
    idx = cuda.grid(1)
    if idx < tensor.shape[0]:
        if xoroshiro128p_uniform_float32(states, idx) < rate:
            noise = tensor.dtype.type(
                xoroshiro128p_normal_float64(states, idx) * strength
            )
            if noise < 0 and low - noise > tensor[idx]:
                tensor[idx] = low
            elif noise > 0 and tensor[idx] + noise > high:
                tensor[idx] = high
            else:
                tensor[idx] += noise


@cuda.jit()
def gaussian_mutation_f32(
    tensor,
    states,
    low: np.number,
    high: np.number,
    rate: np.number,
    strength: np.number,
):
    idx = cuda.grid(1)
    if idx < tensor.shape[0]:
        if xoroshiro128p_uniform_float32(states, idx) < rate:
            noise = xoroshiro128p_normal_float32(states, idx) * strength
            if noise < 0 and low - noise > tensor[idx]:
                tensor[idx] = low
            elif noise > 0 and tensor[idx] + noise > high:
                tensor[idx] = high
            else:
                tensor[idx] += noise


def apply_gaussian(
    tensor: GPUTensor,
    low: np.number,
    high: np.number,
    rate: np.number,
    strength: np.number,
    ctx: dict,
):
    if np.issubdtype(tensor.dtype, np.integer):
        if high is None:
            high = np.iinfo(tensor.dtype).max
        else:
            high -= 1

        if low is None:
            low = np.iinfo(tensor.dtype).min
    else:
        if low is None:
            low = np.finfo(tensor.dtype).min
        if high is None:
            high = np.finfo(tensor.dtype).max

    stream = ctx.get("cuda.stream")

    tensor_flat = tensor.reshape(-1)

    states = create_xoroshiro128p_states(
        tensor_flat.shape[0],
        seed=random.randint(0, 2**32 - 1),
        stream=stream,
    )
    low = tensor.dtype.type(low)
    high = tensor.dtype.type(high)

    threads_per_block = 1024
    blocks_per_grid = (
        tensor_flat.shape[0] + (threads_per_block - 1)
    ) // threads_per_block

    if np.issubdtype(tensor.dtype, np.float32):
        gaussian_mutation_f32[blocks_per_grid, threads_per_block, stream](
            tensor_flat.array, states, low, high, np.float32(rate), np.float32(strength)
        )
    else:
        gaussian_mutation_f64[blocks_per_grid, threads_per_block, stream](
            tensor_flat.array, states, low, high, np.float32(rate), np.float64(strength)
        )
