import time

import numpy as np

from nml.layers import (
    Cast,
    CellularAutomata,
    Flatten,
    LeakyReLU,
    Linear,
    PReLU,
    ReLU,
    Reshape,
    Sigmoid,
    Softmax,
    Tanh,
)
from nml.models import Device, Input, Sequential

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
    LeakyReLU(1),
    Cast(dtype=np.float32),
    Sigmoid(),
    Tanh(),
    Flatten(),
    Linear(784),
    Linear(784, include_bias=False),
    PReLU(),
    Linear(194),
    Linear(10),
    Softmax(),
)

model = test.build()

dataset = np.random.randint(
    low=0,
    high=2,
    size=(60000, 28, 28),
    dtype=np.uint8,
)

print("Testing Softmax on GPU...")
start_gpu = time.time()
gpu_result = model.infer(dataset, device="gpu").wait()
end_gpu = time.time()
print(f"GPU Time: {(end_gpu - start_gpu) * 1000:.2f} ms")

print("Testing Softmax on CPU...")
start_cpu = time.time()
cpu_result = model.infer(dataset, device="cpu").wait()
end_cpu = time.time()
print(f"CPU Time: {(end_cpu - start_cpu) * 1000:.2f} ms")

diff = np.max(np.abs(gpu_result - cpu_result))
print(f"Maximum difference between GPU and CPU results: {diff}")
