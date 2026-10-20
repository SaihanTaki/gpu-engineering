import sys

import numpy as np
from numba import cuda, float32

# Blocks are TILE x TILE threads, one per element of a TILE x TILE tile of C.
TILE = 16


@cuda.jit
def matmul_naive(A, B, C):
    # cuda.grid(2) returns (x, y), and x is the column, as in the tiled kernel.
    col, row = cuda.grid(2)
    n = C.shape[0]
    if row < n and col < n:
        acc = float32(0)
        for k in range(n):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


@cuda.jit
def matmul_tiled(A, B, C):
    # The shape must be a compile-time constant; TILE is a global, so Numba freezes it.
    As = cuda.shared.array((TILE, TILE), float32)
    Bs = cuda.shared.array((TILE, TILE), float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    n = C.shape[0]
    acc = float32(0)
    # No early return for threads past the edge: every thread must reach both barriers.
    for p in range((n + TILE - 1) // TILE):
        a_col = p * TILE + tx
        b_row = p * TILE + ty
        # Zeros past the edge add nothing to the sum, so the inner loop needs no bounds check.
        As[ty, tx] = A[row, a_col] if row < n and a_col < n else float32(0)
        Bs[ty, tx] = B[b_row, col] if b_row < n and col < n else float32(0)
        # The whole tile must be written before any thread reads it.
        cuda.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        # Every thread must be done with this tile before the next phase overwrites it.
        cuda.syncthreads()
    if row < n and col < n:
        C[row, col] = acc


def run(kernel, A, B):
    n = A.shape[0]
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((n, n), dtype=np.float32)
    blocks = (n + TILE - 1) // TILE
    kernel[(blocks, blocks), (TILE, TILE)](d_A, d_B, d_C)
    return d_C.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    B = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    expected = A.astype(np.float64) @ B
    for kernel in (matmul_naive, matmul_tiled):
        np.testing.assert_allclose(run(kernel, A, B), expected, rtol=1e-4, atol=1e-4 * n)
    print("OK")
