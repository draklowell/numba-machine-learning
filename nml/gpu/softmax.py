from numba import cuda

from nml.gpu.tensor import GPUTensor


@cuda.jit(device=True)
def warp_reduce_max(val):
    width = cuda.warpsize
    for offset in range(width // 2, 0, -1):
        other = cuda.shfl_xor_sync(0xFFFFFFFF, val, offset)
        val = max(val, other)
    return val


@cuda.jit(device=True)
def warp_reduce_sum(val):
    width = cuda.warpsize
    for offset in range(width // 2, 0, -1):
        val += cuda.shfl_xor_sync(0xFFFFFFFF, val, offset)
    return val


@cuda.jit(device=True)
def block_reduce_max(val, temp_storage):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    # Reduce the value
    val = warp_reduce_max(val)

    lane = thread_id % 32
    wid = thread_id // 32

    if lane == 0:
        temp_storage[wid] = val

    cuda.syncthreads()

    # Loads reduced values
    if wid == 0:
        val = temp_storage[lane] if lane < (cuda.blockDim.x // 32) else float("-inf")
        val = warp_reduce_max(val)
    return val


@cuda.jit(device=True)
def block_reduce_sum(val, temp_storage):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    # Reduce the value
    val = warp_reduce_max(val)

    lane = thread_id % 32
    wid = thread_id // 32

    if lane == 0:
        temp_storage[wid] = val

    cuda.syncthreads()

    if wid == 0:
        val = temp_storage[lane] if lane < (cuda.blockDim.x // 32) else 0.0
        val = warp_reduce_sum(val)

    return val


@cuda.jit()
def softmax_small_k(x, out, length):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    if thread_id < length:
        val = x[block_id, thread_id]

        # Find max using warp reduction
        local_max = val
        max_val = warp_reduce_max(local_max)

        # Calculate exp and sum
        exp_val = math.exp(val - max_val)
        local_sum = exp_val
        sum_val = warp_reduce_sum(local_sum)

        out[block_id, thread_id] = exp_val / sum_val


@cuda.jit()
def softmax_large_k(x, out, length):
    extern_sm = cuda.shared.array(0, dtype=nb.float32)
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    # Process multiple elements per thread for large K
    local_max = -float("inf")
    for i in range(thread_id, length, cuda.blockDim.x):
        local_max = max(local_max, x[block_id, i])

    # Block reduction for max
    max_val = block_reduce_max(local_max, extern_sm)

    # Broadcast max_val to all threads
    if thread_id == 0:
        extern_sm[0] = max_val
    cuda.syncthreads()
    max_val = extern_sm[0]

    # Calculate exp and sum
    local_sum = 0.0
    for i in range(thread_id, length, cuda.blockDim.x):
        val = math.exp(x[block_id, i] - max_val)
        out[block_id, i] = val
        local_sum += val

    sum_val = block_reduce_sum(local_sum, extern_sm)

    # Broadcast sum_val to all threads
    if thread_id == 0:
        extern_sm[0] = sum_val
    cuda.syncthreads()
    sum_val = extern_sm[0]

    # Normalize
    for i in range(thread_id, length, cuda.blockDim.x):
        out[block_id, i] /= sum_val


def apply_softmax(tensor: GPUTensor, ctx: dict):
    if tensor.ndim != 2:
        raise NotImplementedError("Softmax only supports 2D tensors")

    result = GPUTensor.empty_like(tensor, ctx)
    batch, length = tensor.shape

    # Thresholds based on the article
    k_threshold = 1024

    if length <= k_threshold:
        # For small K, use warp reduction
        threads = min(32, length)
        blocks = batch

        softmax_small_k[blocks, threads, ctx.get("cuda.stream")](
            tensor.array, result.array, length
        )
    else:
        # For large K, use block reduction
        threads = min(1024, length)
        blocks = batch

        # Calculate shared memory size needed - one float per warp
        shared_mem_size = (threads // 32 + 1) * 4

        softmax_large_k[blocks, threads, ctx.get("cuda.stream"), shared_mem_size](
            tensor.array, result.array, length
        )

    return result
