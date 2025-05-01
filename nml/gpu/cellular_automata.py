import numba as nb
import numpy as np
from numba import cuda

from nml.cpu.cellular_automata import compute_mod_table
from nml.gpu.tensor import GPUTensor


@cuda.jit()
def cellular_automata(
    tensor,
    rules,
    neighborhood,
    mod_row,
    mod_col,
    shifts,
    iterations,
    prow,
    pcol,
    mask,
):
    # Get thread indices
    bidx = cuda.blockIdx.x
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Check if indices are within bounds
    if bidx >= tensor.shape[0] or row >= tensor.shape[1] or col >= tensor.shape[2]:
        return

    # Create shared buffers for faster access and write from global to shared
    source = cuda.shared.array((32, 32), dtype=nb.uint8)
    source[row, col] = tensor[bidx, row, col]

    target = cuda.shared.array((32, 32), dtype=nb.uint8)

    # Synchronize copy from global array to shared one
    cuda.syncthreads()

    for _ in range(iterations):
        # Calculate transition index
        transition = nb.uint32(source[row, col])
        for nidx, (nrow, ncol) in enumerate(neighborhood):
            # Use mod lookup tables to handle wrapping
            transition |= (
                nb.uint32(
                    source[
                        mod_row[row + nrow + prow],
                        mod_col[col + ncol + pcol],
                    ]
                    & mask
                )
                << shifts[nidx]
            )

        # Apply the rule
        target[row, col] = rules[transition]

        # Synchronize each iteration
        cuda.syncthreads()

        # Swap buffers
        source, target = target, source

    # Write back to global memory
    tensor[bidx, row, col] = source[row, col]


def build_shifts(
    neighborhood: GPUTensor,
    rule_bitwidth: int,
    ctx: dict | None = None,
) -> GPUTensor:
    if ctx is not None:
        stream = ctx.get("cuda.stream")
    else:
        stream = None

    shifts = np.empty((neighborhood.shape[0],), dtype=np.uint8)
    for nidx in range(neighborhood.shape[0]):
        shifts[nidx] = rule_bitwidth * (nidx + 1)

    shifts = cuda.to_device(shifts, stream=stream)

    return GPUTensor(shifts)


def build_mod_table(
    size: int,
    padding: int,
    ctx: dict | None = None,
) -> GPUTensor:
    if ctx is not None:
        stream = ctx.get("cuda.stream")
    else:
        stream = None

    mod_table = compute_mod_table(size, padding)
    mod_table = cuda.to_device(mod_table, stream=stream)

    return GPUTensor(mod_table)


def apply_cellular_automata(
    tensor: GPUTensor,
    rules: GPUTensor,
    neighborhood: GPUTensor,
    mod_row: GPUTensor,
    mod_col: GPUTensor,
    shifts: GPUTensor,
    prow: int,
    pcol: int,
    iterations: int,
    rule_bitwidth: int,
    ctx: dict,
):
    if tensor.shape[1] > 32 or tensor.shape[2] > 32:
        raise NotImplementedError(
            "GPU cellular automata only supports images with height and width <= 32"
        )

    stream = ctx.get("cuda.stream")

    mask = (1 << rule_bitwidth) - 1

    cellular_automata[tensor.shape[0], tensor.shape[1:], stream](
        tensor.array,
        rules.array,
        neighborhood.array,
        mod_row.array,
        mod_col.array,
        shifts.array,
        np.uint16(iterations),
        prow,
        pcol,
        mask,
    )
    return tensor
