import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tl.arange only accepts power-of-two sizes, so BLOCK must be one.
BLOCK = 1024


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # The last program can cover offsets past the end of the arrays.
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def vector_add(a, b):
    c = torch.empty_like(a)
    n = c.numel()
    vector_add_kernel[(triton.cdiv(n, BLOCK),)](a, b, c, n, BLOCK=BLOCK)
    return c


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def gb_per_s(nbytes, seconds):
    return nbytes / seconds / 1e9


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(n, generator=gen)
    b = torch.rand(n, generator=gen)
    nbytes = a.numel() * a.element_size()

    # The first call compiles the kernel, so it is not timed.
    vector_add(a.to(DEVICE), b.to(DEVICE))
    sync()

    # Launches are asynchronous, so each phase ends with a synchronize.
    t0 = time.perf_counter()
    d_a = a.to(DEVICE)
    d_b = b.to(DEVICE)
    sync()
    t1 = time.perf_counter()
    d_c = vector_add(d_a, d_b)
    sync()
    t2 = time.perf_counter()
    c = d_c.cpu()
    t3 = time.perf_counter()

    torch.testing.assert_close(c, a + b, rtol=0, atol=0)
    print(f"n = {n}, {triton.cdiv(n, BLOCK)} programs of BLOCK = {BLOCK}")
    print(f"  copy in  (a, b):  {(t1 - t0) * 1e3:9.3f} ms  {gb_per_s(2 * nbytes, t1 - t0):8.1f} GB/s")
    print(f"  kernel:           {(t2 - t1) * 1e3:9.3f} ms  {gb_per_s(3 * nbytes, t2 - t1):8.1f} GB/s")
    print(f"  copy out (c):     {(t3 - t2) * 1e3:9.3f} ms  {gb_per_s(nbytes, t3 - t2):8.1f} GB/s")
    print("OK")
