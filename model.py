import time

import numpy as np

start_import = time.time()
import nml

end_import = time.time()

print(f"Import time: {(end_import - start_import) * 1000:.2f} ms")

start_create = time.time()
test = nml.Sequential(
    nml.Input(shape=(28, 28), dtype=np.uint8),
    nml.CellularAutomata(iterations=80),
    nml.CellularAutomata(iterations=80),
    nml.Flatten(),
    nml.Reshape(shape=(28, 28)),
    nml.Cast(dtype=np.uint16),
    nml.Flatten(),
    nml.Reshape(shape=(28, 28)),
    nml.ReLU(),
    nml.Cast(dtype=np.float32),
    nml.Sigmoid(),
    nml.PReLU(),
    nml.LeakyReLU(alpha=0.01),
    nml.Tanh(),
    nml.Flatten(),
    nml.Linear(784),
    nml.Linear(784, include_bias=False),
    nml.Linear(194),
    nml.Linear(10),
    nml.Softmax(),
)
end_create = time.time()

print(f"Model creation time: {(end_create - start_create) * 1000:.2f} ms")

start_build = time.time()
model_cpu = test.build(nml.Device.CPU)
model_gpu = test.build(nml.Device.GPU)
model_gpu.replace_weights(model_cpu.get_weights())
end_build = time.time()

print(f"Model build time: {(end_build - start_build) * 1000:.2f} ms")

dataset = np.random.randint(
    low=0,
    high=2,
    size=(1, 28, 28),
    dtype=np.uint8,
)
dataset_cpu = nml.CPUTensor(dataset)
dataset_gpu = nml.copy_to_device(dataset_cpu, nml.Device.GPU)

print("Testing model on CPU...")
start_cpu = time.time()
cpu_result = model_cpu(dataset_cpu).wait()
end_cpu = time.time()
print(f"CPU Time: {(end_cpu - start_cpu) * 1000:.2f} ms")

print("Testing model on GPU...")
start_gpu = time.time()
gpu_result = model_gpu(dataset_gpu).wait()
end_gpu = time.time()
print(f"GPU Time: {(end_gpu - start_gpu) * 1000:.2f} ms")

gpu_result = nml.copy_to_host(gpu_result)

diff = np.max(np.abs(gpu_result.array - cpu_result.array))
print(f"Maximum difference between GPU and CPU results: {diff}")
