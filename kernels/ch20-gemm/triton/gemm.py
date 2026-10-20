import os
import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configs():
    # The interpreter cannot time candidates, and the autotuner skips timing when there is only one.
    if os.environ.get("TRITON_INTERPRET") == "1":
        return [triton.Config({"BM": 16, "BN": 16, "BK": 16})]
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in [
            (64, 64, 32, 4, 3),
            (128, 64, 32, 4, 3),
            (64, 128, 32, 4, 3),
            (128, 128, 32, 8, 3),
            (128, 128, 16, 8, 4),
            (128, 256, 32, 8, 3),
        ]
    ]


# The first call for each new (M, N, K) times every config and keeps the fastest.
@triton.autotune(configs=configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # One program per BM x BN tile of C; the compiler picks the thread layout and the shared memory staging.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(b_ptr + ks[:, None] * N + rn[None, :], mask=(ks[:, None] < K) & (rn[None, :] < N), other=0.0)
        # "ieee" keeps full float32 on CUDA cores; the default, tf32, would use tensor cores (chapter 23).
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    gemm_kernel[grid](a, b, c, M, N, K)
    return c


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else M
    K = int(sys.argv[3]) if len(sys.argv) > 3 else M
    reps = 10
    g = torch.Generator().manual_seed(0)
    # Whole numbers from -4 to 4: the float32 result is exact, so it must equal torch's.
    a = torch.randint(-4, 5, (M, K), generator=g).float().to(DEVICE)
    b = torch.randint(-4, 5, (K, N), generator=g).float().to(DEVICE)
    if DEVICE == "cuda":
        # Keep torch.matmul on full float32 too, so the two compare like for like.
        torch.backends.cuda.matmul.allow_tf32 = False
    print(f"C = A @ B, M = {M}, N = {N}, K = {K}, float32")
    for name, fn in [("triton", gemm), ("torch", torch.matmul)]:
        # The first call compiles, and for Triton also autotunes, so it is not timed.
        c = fn(a, b)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            c = fn(a, b)
        sync()
        seconds = (time.perf_counter() - t0) / reps
        torch.testing.assert_close(c, a @ b, rtol=0, atol=0)
        print(f"  {name:<7} {seconds * 1e3:9.3f} ms  {2 * M * N * K / seconds / 1e9:8.1f} GFLOP/s")
    print(f"  autotuner picked {gemm_kernel.best_config}")
    print("OK")
