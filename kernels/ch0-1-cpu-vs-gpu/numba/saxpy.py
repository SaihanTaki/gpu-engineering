import sys
import time

import numpy as np
from numba import cuda, njit


@njit
def saxpy_cpu(a, x, y):
    for i in range(y.size):
        y[i] = a * x[i] + y[i]


@cuda.jit
def saxpy(a, x, y):
    i = cuda.grid(1)
    # The last block can have threads past the end of the arrays.
    if i < y.size:
        y[i] = a * x[i] + y[i]


def run_cpu(a, x, y):
    out = y.copy()
    saxpy_cpu(np.float32(a), x, out)
    return out


def run(a, x, y, threads=256):
    blocks = (y.size + threads - 1) // threads
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    saxpy[blocks, threads](np.float32(a), d_x, d_y)
    return d_y.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 24) + 3
    rng = np.random.default_rng(0)
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)

    # The first call of each version compiles it, so only the second is timed.
    run_cpu(2.0, x, y)
    t0 = time.perf_counter()
    expected = run_cpu(2.0, x, y)
    cpu_ms = (time.perf_counter() - t0) * 1e3

    run(2.0, x, y)
    t0 = time.perf_counter()
    result = run(2.0, x, y)
    gpu_ms = (time.perf_counter() - t0) * 1e3

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    print(f"n = {n}")
    print(f"  CPU loop:                          {cpu_ms:9.3f} ms")
    print(f"  GPU copy in, kernel, copy out:     {gpu_ms:9.3f} ms")
    print("OK")
