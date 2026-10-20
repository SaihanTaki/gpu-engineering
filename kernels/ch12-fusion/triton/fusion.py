import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 1024


@triton.jit
def scale_kernel(x_ptr, t_ptr, a, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(t_ptr + offs, a * tl.load(x_ptr + offs, mask=mask), mask=mask)


@triton.jit
def add_kernel(t_ptr, y_ptr, u_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    t = tl.load(t_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(u_ptr + offs, t + y, mask=mask)


@triton.jit
def relu_kernel(u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(u_ptr + offs, mask=mask), 0.0), mask=mask)


@triton.jit
def fused_kernel(x_ptr, y_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    # The whole block of v stays in registers; only out is stored.
    v = a * x + y
    tl.store(out_ptr + offs, tl.maximum(v, 0.0), mask=mask)


def unfused(x, y, a):
    n = x.numel()
    t, u, out = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    scale_kernel[grid](x, t, a, n, BLOCK=BLOCK)
    add_kernel[grid](t, y, u, n, BLOCK=BLOCK)
    relu_kernel[grid](u, out, n, BLOCK=BLOCK)
    return out


def fused(x, y, a):
    n = x.numel()
    out = torch.empty_like(x)
    fused_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, a, n, BLOCK=BLOCK)
    return out


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    a = 1.5
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, generator=g).to(DEVICE)
    y = torch.randn(n, generator=g).to(DEVICE)
    want = torch.relu(a * x + y)

    print(f"n = {n} floats, BLOCK = {BLOCK}")
    for name, run, arrays in [("unfused", unfused, 7), ("fused", fused, 3)]:
        # The first call compiles the kernels, so it is not timed.
        out = run(x, y, a)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = run(x, y, a)
        sync()
        seconds = (time.perf_counter() - t0) / reps
        torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)
        moved = arrays * x.numel() * x.element_size()
        print(f"  {name:<8} {seconds * 1e3:9.4f} ms  {moved / 1e6:7.1f} MB moved  {moved / seconds / 1e9:8.1f} GB/s")
    print("OK")
