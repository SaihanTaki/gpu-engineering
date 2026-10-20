import sys

import numpy as np
from numba import cuda

# A multiple of 32, so every warp of a block is full.
THREADS = 256
# float32 constants keep the arithmetic in float32; a bare 2 or 0.5 would promote it to float64.
TWO = np.float32(2)
ONE = np.float32(1)
HALF = np.float32(0.5)


@cuda.jit
def branchy(x, out):
    i = cuda.grid(1)
    if i < x.size:
        # Lanes of one warp with different signs take different paths.
        if x[i] > 0:
            out[i] = x[i] * TWO + ONE
        else:
            out[i] = x[i] * HALF


@cuda.jit
def branch_free(x, out):
    i = cuda.grid(1)
    if i < x.size:
        v = x[i]
        # Every lane runs the same statements and keeps one of the two results.
        a = v * TWO + ONE
        b = v * HALF
        out[i] = a if v > 0 else b


def run(kernel, x):
    d_x = cuda.to_device(x)
    d_out = cuda.device_array_like(x)
    kernel[(x.size + THREADS - 1) // THREADS, THREADS](d_x, d_out)
    return d_out.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    x = np.random.default_rng(0).standard_normal(n, dtype=np.float32)
    expected = np.where(x > 0, x * TWO + ONE, x * HALF)
    assert np.array_equal(run(branchy, x), expected)
    assert np.array_equal(run(branch_free, x), expected)
    print("OK")
