import sys
import time

import numpy as np
from numba import config, cuda

# numba-cuda has no WMMA or mma API, so these kernels run on the CUDA cores.
# They show the number formats, not tensor-core speed.
BLOCK = 16


@cuda.jit
def gemm_f16_f32acc(a, b, c):
    # FP16 inputs, each product and the sum in FP32. One thread per element of C.
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        acc = np.float32(0.0)
        for k in range(a.shape[1]):
            acc += np.float32(a[i, k]) * np.float32(b[k, j])
        c[i, j] = acc


@cuda.jit
def gemm_f16_f16acc(a, b, c):
    # FP16 inputs and an FP16 running sum: every step rounds to 11 significant bits.
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        acc = np.float16(0.0)
        for k in range(a.shape[1]):
            acc = cuda.fp16.hfma(a[i, k], b[k, j], acc)
        c[i, j] = acc


def run(kernel, a, b, out_dtype):
    c = np.empty((a.shape[0], b.shape[1]), dtype=out_dtype)
    grid = ((c.shape[0] + BLOCK - 1) // BLOCK, (c.shape[1] + BLOCK - 1) // BLOCK)
    d_c = cuda.to_device(c)
    kernel[grid, (BLOCK, BLOCK)](cuda.to_device(a), cuda.to_device(b), d_c)
    return d_c.copy_to_host()


def matmul_f32acc(a, b):
    return run(gemm_f16_f32acc, a, b, np.float32)


def matmul_f16acc(a, b):
    return run(gemm_f16_f16acc, a, b, np.float16)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (n, n)).astype(np.float16)
    b = rng.uniform(-1, 1, (n, n)).astype(np.float16)
    want = a.astype(np.float64) @ b.astype(np.float64)
    scale = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    print(f"C = A B, {n} x {n} x {n}, FP16 inputs" + (" (CUDA simulator)" if config.ENABLE_CUDASIM else ""))
    for name, f in [("FP32 accumulate", matmul_f32acc), ("FP16 accumulate", matmul_f16acc)]:
        t0 = time.perf_counter()
        c = f(a, b)
        s = time.perf_counter() - t0
        err = np.max(np.abs(c.astype(np.float64) - want) / scale)
        print(f"  {name} {s * 1e3:9.1f} ms (first call, includes compile)  max error {err:.2e}")
    # 4096 ones: an FP16 sum stops growing at 2048, where the spacing between FP16 numbers becomes 2.
    ones = np.ones((1, 4096), dtype=np.float16)
    print(f"  sum of 4096 ones: FP32 accumulate {matmul_f32acc(ones, ones.T)[0, 0]:.0f}, FP16 accumulate {matmul_f16acc(ones, ones.T)[0, 0]:.0f}")
    print("OK")
