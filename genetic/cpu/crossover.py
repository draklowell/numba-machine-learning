import random

import numpy as np

from nml import CPUTensor


def _crossover(array1: np.ndarray, array2: np.ndarray, method: str) -> np.ndarray:
    """Helper function to crossover two numpy arrays to produce one child array"""
    flat1 = array1.reshape(-1)
    flat2 = array2.reshape(-1)

    size = flat1.size

    if method == "single_point":
        point = random.randint(1, size - 1)

        flat1[point:] = flat2[point:]
        return array1

    if method == "two_point":
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        flat1[point1:point2] = flat2[point1:point2]
        return array1

    if method == "uniform":
        mask = np.random.rand(size) > 0.5

        flat1[mask] = flat2[mask]
        return array1

    raise ValueError(f"Unknown crossover method: {method}")


def apply_crossover(
    tensor1: CPUTensor,
    tensor2: CPUTensor,
    method: str = "single_point",
) -> CPUTensor:
    """
    Apply crossover to two tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        method: Crossover method. Can be 'single_point', 'two_point', or 'uniform'.

    Returns:
        Offspring tensor.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")

    return CPUTensor(_crossover(tensor1.array, tensor2.array, method))
