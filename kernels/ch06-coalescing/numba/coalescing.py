import sys

import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def copy_offset(src, out, offset):
    i = cuda.grid(1)
    # Lane k reads element k + offset: 5 sectors per warp unless offset is a multiple of 8 floats.
    if i < out.size:
        out[i] = src[i + offset]


@cuda.jit
def copy_strided(src, out, stride):
    i = cuda.grid(1)
    # From stride 8 on, every lane of a warp needs its own 32-byte sector.
    if i < out.size:
        out[i] = src[i * stride]


@cuda.jit
def get_x_aos(p, out):
    i = cuda.grid(1)
    # p is n x 3, one row per particle, so column 0 is a stride-3 read.
    if i < out.size:
        out[i] = p[i, 0]


@cuda.jit
def get_x_soa(x, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = x[i]


def run(kernel, src, n, *args):
    d_src = cuda.to_device(src)
    d_out = cuda.device_array(n, dtype=np.float32)
    kernel[(n + THREADS - 1) // THREADS, THREADS](d_src, d_out, *args)
    return d_out.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 16
    src = np.arange(n * 32, dtype=np.float32)
    for offset in (0, 1, 8):
        assert np.array_equal(run(copy_offset, src, n, offset), src[offset : offset + n])
    for stride in (1, 2, 32):
        assert np.array_equal(run(copy_strided, src, n, stride), src[: n * stride : stride])
    aos = np.random.default_rng(0).standard_normal((n, 3), dtype=np.float32)
    assert np.array_equal(run(get_x_aos, aos, n), aos[:, 0])
    # The SoA layout keeps each field in its own contiguous array.
    soa_x = np.ascontiguousarray(aos[:, 0])
    assert np.array_equal(run(get_x_soa, soa_x, n), soa_x)
    print("OK")
