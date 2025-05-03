import random

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from nml import GPUTensor


@cuda.jit()
def single_point_crossover(
    array1,
    array2,
    point,
):
    idx = cuda.grid(1)
    if idx < array1.shape[0] and idx >= point:
        array1[idx] = array2[idx]


@cuda.jit()
def two_point_crossover(
    array1,
    array2,
    point1,
    point2,
):
    idx = cuda.grid(1)
    if idx < array1.shape[0] and point1 <= idx < point2:
        array1[idx] = array2[idx]


@cuda.jit()
def uniform_crossover(
    array1,
    array2,
    states,
):
    idx = cuda.grid(1)
    if idx < array1.shape[0] and xoroshiro128p_uniform_float32(states, idx) < 0.5:
        array1[idx] = array2[idx]


def apply_crossover(
    tensor1: GPUTensor,
    tensor2: GPUTensor,
    method: str,
    ctx: dict,
) -> GPUTensor:
    """
    Apply crossover to two tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        method: Crossover method. Can be 'single_point', 'two_point', or 'uniform'.
        ctx: Context dictionary for additional information.

    Returns:
        Offspring tensor.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")

    stream = ctx.get("cuda.stream")

    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    size = tensor1_flat.shape[0]

    threads_per_block = 1024
    blocks_per_grid = (size + (threads_per_block - 1)) // threads_per_block
    if method == "single_point":
        point = random.randint(1, size - 1)

        single_point_crossover[blocks_per_grid, threads_per_block, stream](
            tensor1_flat.array, tensor2_flat.array, point
        )
        return tensor1
    elif method == "two_point":
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        two_point_crossover[blocks_per_grid, threads_per_block, stream](
            tensor1_flat.array, tensor2_flat.array, point1, point2
        )
        return tensor1
    elif method == "uniform":
        states = create_xoroshiro128p_states(
            size,
            seed=random.randint(0, 2**32 - 1),
            stream=stream,
        )

        uniform_crossover[blocks_per_grid, threads_per_block, stream](
            tensor1_flat.array, tensor2_flat.array, states
        )
        return tensor1
    else:
        raise ValueError(f"Unknown crossover method: {method}")
