import os
import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tl.dot on BF16 gave wrong results in Triton 3.8's interpreter (errors near 1e10), so BF16 runs only on a GPU.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    # One program per BLOCK_M x BLOCK_N tile of C.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # The accumulator is FP32 whatever the input type.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + rm[:, None] * stride_am + (k + rk)[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + rk)[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        # FP16 or BF16 inputs go to the tensor cores as they are. For FP32 inputs,
        # "tf32" rounds them to TF32 for the tensor cores; "ieee" keeps FP32 on the CUDA cores.
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, input_precision="tf32", block=(64, 64, 32)):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == b.dtype
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    bm, bn, bk = block
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, INPUT_PRECISION=input_precision,
    )
    return c


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    reps = 20
    g = torch.Generator().manual_seed(0)
    a32 = torch.rand((n, n), generator=g) * 2 - 1
    b32 = torch.rand((n, n), generator=g) * 2 - 1
    want = a32.double() @ b32.double()
    # Errors are against the FP64 product of the FP32 inputs, so they include the input rounding.
    print(f"C = A B, {n} x {n} x {n}")
    block = (128, 128, 32) if DEVICE == "cuda" else (32, 32, 32)
    for name, dtype, prec in [("fp16 in, fp32 acc", torch.float16, "ieee"),
                              ("bf16 in, fp32 acc", torch.bfloat16, "ieee"),
                              ("fp32 in, tf32", torch.float32, "tf32"),
                              ("fp32 in, ieee", torch.float32, "ieee")]:
        if dtype == torch.bfloat16 and (INTERPRET or not torch.cuda.is_bf16_supported()):
            print(f"  {name:18s} skipped: {'interpreter' if INTERPRET else 'no BF16 on this GPU'}")
            continue
        a, b = a32.to(dtype).to(DEVICE), b32.to(dtype).to(DEVICE)
        # The first call compiles the kernel, so it is not timed.
        c = matmul(a, b, prec, block)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            c = matmul(a, b, prec, block)
        sync()
        s = (time.perf_counter() - t0) / reps
        err = ((c.cpu().double() - want).abs().max() / want.abs().max()).item()
        print(f"  {name:18s} {s * 1e3:9.3f} ms {2 * n**3 / s / 1e12:8.2f} TFLOPS  max error {err:.1e}")
    print("OK")
