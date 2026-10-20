import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def saxpy(x_ptr, y_ptr, a, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # The last program can cover offsets past the end of the arrays.
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a * x + y, mask=mask)


def run(a, x, y, block=1024):
    # tl.arange only accepts power-of-two sizes, so BLOCK must be one.
    d_x = torch.from_numpy(x).to(DEVICE)
    d_y = torch.from_numpy(y).to(DEVICE, copy=True)
    n = d_y.numel()
    saxpy[(triton.cdiv(n, block),)](d_x, d_y, a, n, BLOCK=block)
    return d_y.cpu().numpy()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 24) + 3
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=gen).numpy()
    y = torch.rand(n, generator=gen).numpy()

    t0 = time.perf_counter()
    expected = 2.0 * x + y
    cpu_ms = (time.perf_counter() - t0) * 1e3

    # The first call compiles the kernel, so only the second is timed.
    run(2.0, x, y)
    t0 = time.perf_counter()
    result = run(2.0, x, y)
    gpu_ms = (time.perf_counter() - t0) * 1e3

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    print(f"n = {n}")
    print(f"  CPU (NumPy):                       {cpu_ms:9.3f} ms")
    print(f"  GPU copy in, kernel, copy out:     {gpu_ms:9.3f} ms")
    print("OK")
