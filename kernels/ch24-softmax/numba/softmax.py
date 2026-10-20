import math
import sys
import time

import numpy as np
from numba import cuda, float32

# Threads per block, a power of two for the shared memory tree.
BLOCK = 128
NEG_INF = float32(-math.inf)


# Chapter 13's sequential-addressing tree, so the kernels also run on the
# Numba simulator, which has no warp shuffles. Every thread gets the result.
# Device functions are inlined in Numba's IR, which lets compile_ptx build the
# kernels on a machine without a GPU.
@cuda.jit(device=True, inline="always")
def block_max(v, sm):
    tid = cuda.threadIdx.x
    cuda.syncthreads()  # sm may still be read by an earlier call
    sm[tid] = v
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            sm[tid] = max(sm[tid], sm[tid + s])
        cuda.syncthreads()
        s //= 2
    return sm[0]


@cuda.jit(device=True, inline="always")
def block_sum(v, sm):
    tid = cuda.threadIdx.x
    cuda.syncthreads()
    sm[tid] = v
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            sm[tid] += sm[tid + s]
        cuda.syncthreads()
        s //= 2
    return sm[0]


@cuda.jit
def softmax_safe(x, y):
    # One block per row: max pass, sum pass, normalize pass.
    sm = cuda.shared.array(BLOCK, float32)
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    cols = x.shape[1]
    m = NEG_INF
    for i in range(tid, cols, BLOCK):
        m = max(m, x[r, i])
    m = block_max(m, sm)
    d = float32(0)
    for i in range(tid, cols, BLOCK):
        d += math.exp(x[r, i] - m)
    d = block_sum(d, sm)
    for i in range(tid, cols, BLOCK):
        y[r, i] = math.exp(x[r, i] - m) / d


@cuda.jit(device=True, inline="always")
def md_merge(m1, d1, m2, d2):
    # Milakov and Gimelshein's merge: rescale both sums to the larger max, then add.
    m = max(m1, m2)
    if m == NEG_INF:
        return m, float32(0)
    return m, d1 * math.exp(m1 - m) + d2 * math.exp(m2 - m)


@cuda.jit
def softmax_online(x, y):
    # One block per row: m and d in one pass, then normalize.
    sm_m = cuda.shared.array(BLOCK, float32)
    sm_d = cuda.shared.array(BLOCK, float32)
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    cols = x.shape[1]
    m = NEG_INF
    d = float32(0)
    for i in range(tid, cols, BLOCK):
        m, d = md_merge(m, d, x[r, i], float32(1))
    sm_m[tid] = m
    sm_d[tid] = d
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            sm_m[tid], sm_d[tid] = md_merge(sm_m[tid], sm_d[tid], sm_m[tid + s], sm_d[tid + s])
        cuda.syncthreads()
        s //= 2
    m = sm_m[0]
    d = sm_d[0]
    for i in range(tid, cols, BLOCK):
        y[r, i] = math.exp(x[r, i] - m) / d


@cuda.jit(device=True, inline="always")
def welford_merge(n1, mean1, m21, n2, mean2, m22):
    # Chan, Golub and LeVeque's pairwise merge of count, mean and M2.
    n = n1 + n2
    if n == 0:
        return n1, mean1, m21
    delta = mean2 - mean1
    wb = n2 / n
    return n, mean1 + delta * wb, m21 + m22 + delta * delta * n1 * wb


@cuda.jit
def layernorm_welford(x, gamma, beta, y, mean_out, rstd_out, eps):
    sm_n = cuda.shared.array(BLOCK, float32)
    sm_mean = cuda.shared.array(BLOCK, float32)
    sm_m2 = cuda.shared.array(BLOCK, float32)
    r = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    cols = x.shape[1]
    n = float32(0)
    mean = float32(0)
    m2 = float32(0)
    for i in range(tid, cols, BLOCK):
        v = x[r, i]
        n += float32(1)
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)
    sm_n[tid] = n
    sm_mean[tid] = mean
    sm_m2[tid] = m2
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            sm_n[tid], sm_mean[tid], sm_m2[tid] = welford_merge(
                sm_n[tid], sm_mean[tid], sm_m2[tid], sm_n[tid + s], sm_mean[tid + s], sm_m2[tid + s]
            )
        cuda.syncthreads()
        s //= 2
    mean = sm_mean[0]
    rstd = float32(1) / math.sqrt(sm_m2[0] / cols + eps)
    for i in range(tid, cols, BLOCK):
        y[r, i] = (x[r, i] - mean) * rstd * gamma[i] + beta[i]
    if tid == 0:
        mean_out[r] = mean
        rstd_out[r] = rstd


SOFTMAX = {"safe": softmax_safe, "online": softmax_online}


def softmax(x, method="online"):
    d_x = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    d_y = cuda.device_array_like(d_x)
    SOFTMAX[method][x.shape[0], BLOCK](d_x, d_y)
    return d_y.copy_to_host()


def layernorm(x, gamma, beta, eps=1e-5):
    rows = x.shape[0]
    d_x = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    d_y = cuda.device_array_like(d_x)
    d_mean = cuda.device_array(rows, dtype=np.float32)
    d_rstd = cuda.device_array(rows, dtype=np.float32)
    layernorm_welford[rows, BLOCK](d_x, cuda.to_device(gamma), cuda.to_device(beta), d_y, d_mean, d_rstd, np.float32(eps))
    return d_y.copy_to_host(), d_mean.copy_to_host(), d_rstd.copy_to_host()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    rng = np.random.default_rng(0)
    x = (3 * rng.standard_normal((rows, cols))).astype(np.float32)
    x[0, cols // 2] = 1000.0
    d_x = cuda.to_device(x)
    d_y = cuda.device_array_like(d_x)
    reps = 20
    # Effective bandwidth counts the row read once and written once.
    print(f"rows = {rows}, cols = {cols}, {BLOCK} threads per block")
    for name, kernel in SOFTMAX.items():
        # The first call compiles the kernel, so it is not timed.
        kernel[rows, BLOCK](d_x, d_y)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            kernel[rows, BLOCK](d_x, d_y)
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        y = d_y.copy_to_host()
        assert np.isfinite(y).all() and np.allclose(y.sum(axis=1), 1, atol=1e-4), name
        print(f"  softmax_{name:<7} {seconds * 1e3:9.3f} ms  {8 * rows * cols / seconds / 1e9:8.1f} GB/s")
    print("OK")
