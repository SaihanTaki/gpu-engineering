import sys
import time

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    # The last block can have threads past the end of the arrays.
    if i < c.size:
        c[i] = a[i] + b[i]


def run(a, b):
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)
    blocks = (a.size + THREADS - 1) // THREADS
    vector_add[blocks, THREADS](d_a, d_b, d_c)
    return d_c.copy_to_host()


def gb_per_s(nbytes, seconds):
    return nbytes / seconds / 1e9


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)

    # The first call compiles the kernel, so it is not timed.
    run(a, b)

    # Launches are asynchronous, so each phase ends with a synchronize.
    t0 = time.perf_counter()
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)
    cuda.synchronize()
    t1 = time.perf_counter()
    blocks = (n + THREADS - 1) // THREADS
    vector_add[blocks, THREADS](d_a, d_b, d_c)
    cuda.synchronize()
    t2 = time.perf_counter()
    c = d_c.copy_to_host()
    t3 = time.perf_counter()

    np.testing.assert_array_equal(c, a + b)
    print(f"n = {n}, {blocks} blocks of {THREADS} threads")
    print(f"  copy in  (a, b):  {(t1 - t0) * 1e3:9.3f} ms  {gb_per_s(2 * a.nbytes, t1 - t0):8.1f} GB/s")
    print(f"  kernel:           {(t2 - t1) * 1e3:9.3f} ms  {gb_per_s(3 * a.nbytes, t2 - t1):8.1f} GB/s")
    print(f"  copy out (c):     {(t3 - t2) * 1e3:9.3f} ms  {gb_per_s(a.nbytes, t3 - t2):8.1f} GB/s")
    print("OK")
