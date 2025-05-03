import numpy as np

from nml import CPUTensor, Scalar, Tensor


def mutate_integer(
    tensor: CPUTensor,
    min_value: np.number,
    max_value: np.number,
    rate: np.number,
    strength: np.number,
) -> CPUTensor:
    unsigned_to_signed_map = {
        np.uint8: np.int16,
        np.uint16: np.int32,
        np.uint32: np.int64,
    }

    dtype = tensor.dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        if dtype.type not in unsigned_to_signed_map:
            raise ValueError(f"Unsupported unsigned integer type: {dtype.type}")

        tensor = tensor.cast(unsigned_to_signed_map[tensor.dtype.type])

    if max_value is None:
        max_value = np.iinfo(dtype).max
    else:
        max_value -= 1
    mask = np.random.random(tensor.shape) < rate
    noise = np.random.normal(0, strength, tensor.shape).astype(tensor.dtype)
    tensor.array[mask] += noise[mask]
    tensor.array = np.clip(tensor.array, min_value, max_value)

    return tensor.cast(dtype)


def mutate_float(
    tensor: CPUTensor,
    low: np.number,
    max_value: np.number,
    min_value: np.number,
    strength: np.number,
) -> CPUTensor:
    mask = np.random.random(tensor.shape) < min_value
    noise = np.random.normal(0, strength, tensor.shape)
    tensor.array[mask] += noise[mask]
    tensor.array = np.clip(tensor.array, low, max_value)

    return tensor


def mutate_scalar(
    tensor: Scalar,
    min_value: np.number,
    max_value: np.number,
    rate: np.number,
    strength: np.number,
) -> Scalar:
    if np.random.random() >= rate:
        return tensor

    noise = np.random.normal(0, strength)
    if noise < 0 and min_value - noise > tensor.value:
        tensor.value = min_value
    else:
        tensor.value += noise
        tensor.value = np.clip(tensor.value, None, max_value)

    return tensor


def apply_gaussian(
    tensor: Tensor,
    low: np.number,
    high: np.number,
    rate: np.number,
    strength: np.number,
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

    if tensor.device is None:
        return mutate_scalar(tensor, low, high, rate, strength)

    if np.issubdtype(tensor.dtype, np.integer):
        return mutate_integer(tensor, low, high, rate, strength)

    return mutate_float(tensor, low, high, rate, strength)
