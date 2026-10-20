import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BINS = 256
BLOCK = 1024


@triton.jit
def hist_atomic_kernel(data_ptr, hist_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    b = tl.load(data_ptr + offs, mask=mask, other=0).to(tl.int32)
    # One atomic per element, like the CUDA global kernel.
    tl.atomic_add(hist_ptr + b, 1, mask=mask, sem="relaxed")


@triton.jit
def hist_local_kernel(data_ptr, hist_ptr, n, BINS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    b = tl.load(data_ptr + offs, mask=mask, other=0).to(tl.int32)
    # The program's own histogram; the compiler decides how to build it.
    counts = tl.histogram(b, BINS, mask=mask)
    # One atomic per bin per program.
    tl.atomic_add(hist_ptr + tl.arange(0, BINS), counts, sem="relaxed")


def histogram(data, variant="local"):
    n = data.numel()
    hist = torch.zeros(BINS, dtype=torch.int32, device=data.device)
    grid = (triton.cdiv(n, BLOCK),)
    if variant == "atomic":
        hist_atomic_kernel[grid](data, hist, n, BLOCK=BLOCK)
    else:
        hist_local_kernel[grid](data, hist, n, BINS=BINS, BLOCK=BLOCK)
    return hist


KERNELS = ("atomic", "local")


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    rng = np.random.default_rng(0)
    inputs = {
        "uniform": rng.integers(0, BINS, n, dtype=np.uint8),
        "skewed": np.minimum(rng.geometric(0.5, n) - 1, BINS - 1).astype(np.uint8),
        "one-bin": np.full(n, 7, dtype=np.uint8),
    }
    print(f"{n} bytes, BLOCK = {BLOCK}")
    for dist, data in inputs.items():
        ref = np.bincount(data, minlength=BINS)
        x = torch.from_numpy(data).to(DEVICE)
        for variant in KERNELS:
            # The first call compiles the kernel, so it is not timed.
            out = histogram(x, variant)
            sync()
            np.testing.assert_array_equal(out.cpu().numpy(), ref)
            t0 = time.perf_counter()
            for _ in range(reps):
                out = histogram(x, variant)
            sync()
            seconds = (time.perf_counter() - t0) / reps
            print(f"  {dist:<8} {variant:<7} {seconds * 1e3:9.3f} ms  {n / seconds / 1e9:8.1f} GB/s")
    print("OK")
