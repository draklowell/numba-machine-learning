import numpy as np

_MOORE_1 = []
_MOORE_2 = []
_VON_NEUMANN_1 = []
_VON_NEUMANN_2 = []

for i in range(-2, 3):
    for j in range(-2, 3):
        if 0 < abs(i) + abs(j) <= 2:
            _VON_NEUMANN_2.append((i, j))
        if 0 < abs(i) + abs(j) <= 1:
            _VON_NEUMANN_1.append((i, j))

        if 0 < max(abs(i), abs(j)) <= 2:
            _MOORE_2.append((i, j))
        if 0 < max(abs(i), abs(j)) <= 1:
            _MOORE_1.append((i, j))

MOORE_1 = np.array(_MOORE_1, dtype=np.int8)
MOORE_2 = np.array(_MOORE_2, dtype=np.int8)
VON_NEUMANN_1 = np.array(_VON_NEUMANN_1, dtype=np.int8)
VON_NEUMANN_2 = np.array(_VON_NEUMANN_2, dtype=np.int8)

NEIGHBORHOODS = {
    "moore_1": MOORE_1,
    "moore_2": MOORE_2,
    "von_neumann_1": VON_NEUMANN_1,
    "von_neumann_2": VON_NEUMANN_2,
}
