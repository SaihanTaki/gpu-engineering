import sys
import time

import numpy as np
from numba import cuda, float32

# C = A @ B for row-major float32 matrices: A is M x K, B is K x N.


@cuda.jit
def gemm_naive(A, B, C):
    # cuda.grid(2) is (x, y) and x is the fastest-changing index across a warp, so lanes walk columns: coalesced.
    col, row = cuda.grid(2)
    M, K = A.shape
    N = B.shape[1]
    if row < M and col < N:
        acc = float32(0)
        for k in range(K):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


# Shared memory tiling: TILE x TILE threads per block, one element of C each.
TILE = 16


@cuda.jit
def gemm_tiled(A, B, C):
    As = cuda.shared.array((TILE, TILE), float32)
    Bs = cuda.shared.array((TILE, TILE), float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    M, K = A.shape
    N = B.shape[1]
    acc = float32(0)
    # No early return past the edge: every thread must reach both barriers.
    for k0 in range(0, K, TILE):
        As[ty, tx] = A[row, k0 + tx] if row < M and k0 + tx < K else float32(0)
        Bs[ty, tx] = B[k0 + ty, col] if k0 + ty < K and col < N else float32(0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        cuda.syncthreads()
    if row < M and col < N:
        C[row, col] = acc


def make_regblock(BM, BN, BK, TM, TN):
    # Shared and local array shapes must be compile-time constants, so each tile shape gets its own kernel.
    THREADS = (BM // TM) * (BN // TN)

    @cuda.jit
    def gemm_regblock(A, B, C):
        As = cuda.shared.array((BM, BK), float32)
        Bs = cuda.shared.array((BK, BN), float32)
        acc = cuda.local.array((TM, TN), float32)
        a = cuda.local.array(TM, float32)
        b = cuda.local.array(TN, float32)
        M, K = A.shape
        N = B.shape[1]
        tid = cuda.threadIdx.x
        tr = tid // (BN // TN)
        tc = tid % (BN // TN)
        row0 = cuda.blockIdx.y * BM
        col0 = cuda.blockIdx.x * BN
        for i in range(TM):
            for j in range(TN):
                acc[i, j] = 0
        for k0 in range(0, K, BK):
            # Consecutive threads load consecutive columns: coalesced. Zeros past the edge add nothing.
            for e in range(tid, BM * BK, THREADS):
                r = row0 + e // BK
                c = k0 + e % BK
                As[e // BK, e % BK] = A[r, c] if r < M and c < K else float32(0)
            for e in range(tid, BK * BN, THREADS):
                r = k0 + e // BN
                c = col0 + e % BN
                Bs[e // BN, e % BN] = B[r, c] if r < K and c < N else float32(0)
            cuda.syncthreads()
            for k in range(BK):
                # TM + TN shared memory reads feed TM * TN multiply-adds.
                for i in range(TM):
                    a[i] = As[tr * TM + i, k]
                for j in range(TN):
                    b[j] = Bs[k, tc * TN + j]
                for i in range(TM):
                    for j in range(TN):
                        acc[i, j] += a[i] * b[j]
            cuda.syncthreads()
        for i in range(TM):
            for j in range(TN):
                r = row0 + tr * TM + i
                c = col0 + tc * TN + j
                if r < M and c < N:
                    C[r, c] = acc[i, j]

    gemm_regblock.tile = (BM, BN, THREADS)
    return gemm_regblock


gemm_regblock = make_regblock(64, 64, 8, 4, 4)

KERNELS = {"naive": gemm_naive, "tiled": gemm_tiled, "regblock": gemm_regblock}


def launch(name, d_A, d_B, d_C):
    M, N = d_C.shape
    if name == "regblock":
        BM, BN, threads = gemm_regblock.tile
        gemm_regblock[((N + BN - 1) // BN, (M + BM - 1) // BM), threads](d_A, d_B, d_C)
    else:
        size = 32 if name == "naive" else TILE
        KERNELS[name][((N + size - 1) // size, (M + size - 1) // size), (size, size)](d_A, d_B, d_C)


def run(name, A, B):
    d_A = cuda.to_device(np.ascontiguousarray(A, dtype=np.float32))
    d_B = cuda.to_device(np.ascontiguousarray(B, dtype=np.float32))
    d_C = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)
    launch(name, d_A, d_B, d_C)
    return d_C.copy_to_host()


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    N = int(sys.argv[2]) if len(sys.argv) > 2 else M
    K = int(sys.argv[3]) if len(sys.argv) > 3 else M
    reps = 10
    rng = np.random.default_rng(0)
    # Whole numbers from -4 to 4: the float32 result is exact, so it must equal NumPy's.
    A = rng.integers(-4, 5, (M, K)).astype(np.float32)
    B = rng.integers(-4, 5, (K, N)).astype(np.float32)
    ref = A @ B
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    print(f"C = A @ B, M = {M}, N = {N}, K = {K}, float32")
    for name in KERNELS:
        # The first call compiles the kernel, so it is not timed.
        launch(name, d_A, d_B, d_C)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            launch(name, d_A, d_B, d_C)
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        np.testing.assert_array_equal(d_C.copy_to_host(), ref)
        print(f"  {name:<9} {seconds * 1e3:9.3f} ms  {2 * M * N * K / seconds / 1e9:8.1f} GFLOP/s")
    print("OK")
