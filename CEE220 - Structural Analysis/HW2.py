import numpy as np
import time

# Define full Bf matrix (7x7)
Bf_full = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [-1/10, -1/10, 1/10, 1/10, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, -1/5, -1/5, 1/10],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
], dtype=float)

# Define reduced Bf matrix (4x4, without trivial DOF)
Bf_reduced = np.array([
    [-1/10, 1/10, 1/10, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1/10]
], dtype=float)

# 1M repetitions.
N = 1000000

# Measure inversion time for full matrix
start = time.time()
for _ in range(N):
    np.linalg.inv(Bf_full)
time_full = time.time() - start

# Measure inversion time for reduced matrix
start = time.time()
for _ in range(N):
    np.linalg.inv(Bf_reduced)
time_reduced = time.time() - start

print(f"Time for {N} inversions (full 7x7 matrix): {time_full:.6f} seconds")
print(f"Time for {N} inversions (reduced 4x4 matrix): {time_reduced:.6f}  seconds")
print(f"Reduction of {round((time_full-time_reduced)/time_reduced*100,2)}%")
