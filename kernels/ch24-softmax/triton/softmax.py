import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    # One program per row, and the whole row fits in one block: read once, write once.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    # Padding reads as -inf, so it never wins the max and its exp is 0.
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(y_ptr + row * stride + offs, num / tl.sum(num, axis=0), mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    # The row is on chip, so the two-pass variance costs no extra memory traffic.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + row * stride + offs, xc * rstd * w + b, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def block_for(n_cols):
    block = triton.next_power_of_2(n_cols)
    # Longer rows need a loop over blocks, as in kernels/ch24-softmax/cuda.
    assert block <= 65536, "row too long for one block"
    return block


def softmax(x):
    y = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](x, y, x.stride(0), x.shape[1], BLOCK=block_for(x.shape[1]))
    return y


def layernorm(x, w, b, eps=1e-5):
    rows, cols = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    layernorm_kernel[(rows,)](x, y, w, b, mean, rstd, x.stride(0), cols, eps, BLOCK=block_for(cols))
    return y, mean, rstd


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    g = torch.Generator().manual_seed(0)
    x = 3 * torch.randn(rows, cols, generator=g)
    x[0, cols // 2] = 1000.0
    x = x.to(DEVICE)
    w = torch.ones(cols, device=DEVICE)
    b = torch.zeros(cols, device=DEVICE)
    reps = 20
    # Effective bandwidth counts the row read once and written once.
    print(f"rows = {rows}, cols = {cols}, BLOCK = {block_for(cols)}")
    for name, fn in {"softmax": lambda: softmax(x), "layernorm": lambda: layernorm(x, w, b)}.items():
        # The first call compiles the kernel, so it is not timed.
        fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        seconds = (time.perf_counter() - t0) / reps
        print(f"  {name:<9} {seconds * 1e3:9.3f} ms  {8 * rows * cols / seconds / 1e9:8.1f} GB/s")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))
    print("OK")
