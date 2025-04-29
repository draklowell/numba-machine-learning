import time

import numpy as np

start_import = time.time()

from nml import (
    Cast,
    CellularAutomata,
    CPUTensor,
    Device,
    Flatten,
    GPUTensor,
    Input,
    LeakyReLU,
    Linear,
    PReLU,
    ReLU,
    Reshape,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)

end_import = time.time()

print(f"Import time: {(end_import - start_import) * 1000:.2f} ms")

start_create = time.time()

test = Sequential(
    Input(shape=(28, 28), dtype=np.uint8),
    CellularAutomata(iterations=80),
    CellularAutomata(iterations=80),
    Flatten(),
    Reshape(shape=(28, 28)),
    Cast(dtype=np.uint16),
    Flatten(),
    Reshape(shape=(28, 28)),
    ReLU(),
    Cast(dtype=np.float32),
    Sigmoid(),
    PReLU(),
    LeakyReLU(alpha=0.01),
    Tanh(),
    Flatten(),
    Linear(784),
    Linear(784, include_bias=False),
    Linear(194),
    Linear(10),
    Softmax(),
)

end_create = time.time()

print(f"Model creation time: {(end_create - start_create) * 1000:.2f} ms")

start_build = time.time()

model_cpu = test.build(Device.CPU)
model_gpu = test.build(Device.GPU)

end_build = time.time()

print(f"Model build time: {(end_build - start_build) * 1000:.2f} ms")

dataset = np.random.randint(
    low=0,
    high=2,
    size=(1, 28, 28),
    dtype=np.uint8,
)
dataset_cpu = CPUTensor.create(dataset, dtype=np.uint8)
dataset_gpu = GPUTensor.create(dataset, dtype=np.uint8)

print("Testing Softmax on CPU...")
start_cpu = time.time()
cpu_result = model_cpu(dataset_cpu).wait()
end_cpu = time.time()
print(f"CPU Time: {(end_cpu - start_cpu) * 1000:.2f} ms")

print("Testing Softmax on GPU...")
start_gpu = time.time()
gpu_result = model_gpu(dataset_gpu).wait()
end_gpu = time.time()
print(f"GPU Time: {(end_gpu - start_gpu) * 1000:.2f} ms")

diff = np.max(np.abs(gpu_result - cpu_result))
print(f"Maximum difference between GPU and CPU results: {diff}")
