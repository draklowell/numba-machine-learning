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
from nml.models import Input, Sequential

np.random.seed(42)

conway_rules = np.zeros((512), dtype=np.uint8)
for idx in range(256):
    idx <<= 1
    conway_rules[idx] = 1 if idx.bit_count() == 3 else 0
    conway_rules[idx | 1] = 1 if idx.bit_count() in {2, 3} else 0

test = Sequential(
    Input(shape=(28, 28), dtype=np.uint8),
    CellularAutomata(iterations=3),
    CellularAutomata(iterations=3),
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
    Linear(194),
    Linear(10),
    Tanh(),
)
model = test.build()
model.set_weights(
    {
        "cellular_automata_1": {
            "rules": conway_rules,
        }
    },
    update=True,
)

value = np.random.randint(
    low=0,
    high=2,
    size=(60000, 28, 28),
    dtype=np.uint8,
)
# res_gpu = model.infer(value.copy(), device="gpu").wait()
# res_cpu = model.infer(value.copy(), device="cpu").wait()
# print("ORIGINAL")
# print(value)
# print("MASK")
# print(np.isclose(res_cpu, res_gpu))
# # print(res_cpu res_gpu))
# print("CPU")
# print(res_cpu)
# print("GPU")
# print(res_gpu)
# print(res_gpu.dtype)
# print(res_gpu.shape)

start = time.time()
res_gpu = model.infer(value, device="gpu").wait()
end = time.time()
print(end - start)
