import os
import sys

import numpy as np
from numba import cuda, int32

DEGREE = 8
BINS = 16
KEYS_PER_THREAD = 64
THREADS = 256
# Read when the kernel compiles and baked into constant memory, so changing it later has no effect.
COEF = (1.0 / np.arange(1, DEGREE + 2)).astype(np.float32)


@cuda.jit
def poly_constant(x, y):
    i = cuda.grid(1)
    if i < x.size:
        c = cuda.const.array_like(COEF)
        v = x[i]
        acc = c[DEGREE]
        for k in range(DEGREE - 1, -1, -1):
            acc = acc * v + c[k]
        y[i] = acc


@cuda.jit
def bin_counts(keys, counts):
    t = cuda.grid(1)
    if t < counts.shape[0]:
        # Private to this thread; indexed by data, so it lives in local memory.
        bins = cuda.local.array(BINS, int32)
        for b in range(BINS):
            bins[b] = 0
        for k in range(KEYS_PER_THREAD):
            bins[keys[t * KEYS_PER_THREAD + k] % BINS] += 1
        for b in range(BINS):
            counts[t, b] = bins[b]


def run_poly(x):
    d_x = cuda.to_device(x)
    d_y = cuda.device_array_like(x)
    poly_constant[(x.size + THREADS - 1) // THREADS, THREADS](d_x, d_y)
    return d_y.copy_to_host()


def run_bins(keys):
    threads = keys.size // KEYS_PER_THREAD
    d_counts = cuda.device_array((threads, BINS), dtype=np.int32)
    bin_counts[(threads + THREADS - 1) // THREADS, THREADS](cuda.to_device(keys), d_counts)
    return d_counts.copy_to_host()


def poly_reference(x):
    acc = np.full_like(x, COEF[DEGREE])
    for k in range(DEGREE - 1, -1, -1):
        acc = acc * x + COEF[k]
    return acc


def bins_reference(keys):
    per_thread = keys.reshape(-1, KEYS_PER_THREAD) % BINS
    return np.stack([np.bincount(row, minlength=BINS) for row in per_thread]).astype(np.int32)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 16
    x = np.sin(0.001 * np.arange(n, dtype=np.float32))
    np.testing.assert_allclose(run_poly(x), poly_reference(x), rtol=1e-5)
    keys = np.random.default_rng(0).integers(0, 1 << 20, 4096 * KEYS_PER_THREAD, dtype=np.int32)
    assert np.array_equal(run_bins(keys), bins_reference(keys))
    if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
        for kernel in (poly_constant, bin_counts):
            # Lazily compiled kernels report {signature: value} for each compiled version.
            regs, local = kernel.get_regs_per_thread(), kernel.get_local_mem_per_thread()
            for sig in regs:
                print(f"{kernel.__name__}{sig}: {regs[sig]} registers, {local[sig]} bytes local memory per thread")
    else:
        print("Simulator: no register or local memory counts.")
    print("OK")
