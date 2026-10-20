import sys
import time

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def scale(x, t, a):
    i = cuda.grid(1)
    if i < x.size:
        t[i] = a * x[i]


@cuda.jit
def add(t, y, u):
    i = cuda.grid(1)
    if i < t.size:
        u[i] = t[i] + y[i]


@cuda.jit
def relu(u, out):
    i = cuda.grid(1)
    if i < u.size:
        out[i] = max(u[i], np.float32(0.0))


@cuda.jit
def fused(x, y, out, a):
    i = cuda.grid(1)
    if i < x.size:
        # v never leaves a register, so t and u never exist in DRAM.
        v = a * x[i] + y[i]
        out[i] = max(v, np.float32(0.0))


def blocks(n):
    return (n + THREADS - 1) // THREADS


def run_unfused(d_x, d_y, d_t, d_u, d_out, a):
    g = blocks(d_x.size)
    scale[g, THREADS](d_x, d_t, a)
    add[g, THREADS](d_t, d_y, d_u)
    relu[g, THREADS](d_u, d_out)


def run_fused(d_x, d_y, d_out, a):
    fused[blocks(d_x.size), THREADS](d_x, d_y, d_out, a)


def unfused_host(x, y, a):
    d_x, d_y = cuda.to_device(x), cuda.to_device(y)
    d_t, d_u, d_out = (cuda.device_array_like(x) for _ in range(3))
    run_unfused(d_x, d_y, d_t, d_u, d_out, np.float32(a))
    return d_out.copy_to_host()


def fused_host(x, y, a):
    d_x, d_y = cuda.to_device(x), cuda.to_device(y)
    d_out = cuda.device_array_like(x)
    run_fused(d_x, d_y, d_out, np.float32(a))
    return d_out.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    a = np.float32(1.5)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n, dtype=np.float32)
    y = rng.standard_normal(n, dtype=np.float32)
    d_x, d_y = cuda.to_device(x), cuda.to_device(y)
    d_t, d_u, d_out = (cuda.device_array_like(x) for _ in range(3))
    want = np.maximum(a * x + y, 0)

    print(f"n = {n} floats, {THREADS} threads per block")
    for name, run, arrays in [
        ("unfused", lambda: run_unfused(d_x, d_y, d_t, d_u, d_out, a), 7),
        ("fused", lambda: run_fused(d_x, d_y, d_out, a), 3),
    ]:
        # The first call compiles the kernels, so it is not timed.
        run()
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            run()
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        np.testing.assert_allclose(d_out.copy_to_host(), want, rtol=1e-5, atol=1e-5)
        moved = arrays * x.nbytes
        print(f"  {name:<8} {seconds * 1e3:9.4f} ms  {moved / 1e6:7.1f} MB moved  {moved / seconds / 1e9:8.1f} GB/s")
    print("OK")
