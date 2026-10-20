import sys
import time

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def scale(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = 2.0 * x[i]


def blocks_for(n):
    return (n + THREADS - 1) // THREADS


def run_pageable(x):
    # x is an ordinary NumPy array, so the driver stages both copies.
    d_x = cuda.to_device(x)
    scale[blocks_for(x.size), THREADS](d_x)
    return d_x.copy_to_host()


def run_pinned(x):
    h_x = cuda.pinned_array(x.shape, dtype=x.dtype)
    h_x[:] = x
    d_x = cuda.to_device(h_x)
    scale[blocks_for(x.size), THREADS](d_x)
    d_x.copy_to_host(h_x)
    return h_x


def run_mapped(x):
    # Pinned host memory the kernel reads and writes over PCIe; nothing is copied.
    m = cuda.mapped_array(x.shape, dtype=x.dtype)
    m[:] = x
    scale[blocks_for(x.size), THREADS](m)
    cuda.synchronize()
    return np.array(m)


def run_managed(x):
    # One buffer for both sides; pages fault over on first touch.
    m = cuda.managed_array(x.shape, dtype=x.dtype)
    m[:] = x
    scale[blocks_for(x.size), THREADS](m)
    # The kernel is asynchronous and the CPU must not read m while it runs.
    cuda.synchronize()
    return np.array(m)


RUNS = {"pageable": run_pageable, "pinned": run_pinned, "mapped": run_mapped, "managed": run_managed}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 26
    x = (np.arange(n) % 1000).astype(np.float32)
    print(f"n = {n} floats ({x.nbytes / 2**20:.1f} MiB)")
    for name, fn in RUNS.items():
        if name == "managed" and not hasattr(cuda, "managed_array"):
            print(f"  {name:9s} skipped: the CUDA simulator has no managed_array")
            continue
        # The first call compiles the kernel, so it is not timed.
        fn(x[:THREADS])
        t0 = time.perf_counter()
        y = fn(x)
        cuda.synchronize()
        t1 = time.perf_counter()
        np.testing.assert_array_equal(y, 2 * x)
        # Allocation time is included; it is part of the cost of each choice.
        print(f"  {name:9s} {(t1 - t0) * 1e3:9.3f} ms  {2 * x.nbytes / (t1 - t0) / 1e9:7.2f} GB/s")
    print("OK")
