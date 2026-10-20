import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keys per program. tl.sort works on a whole block, and tl.arange needs a power of two.
BLOCK = 1024


@triton.jit
def block_sort_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # Missing keys of the last block load as +inf, so they sort to the end and are not stored.
    keys = tl.load(x_ptr + offs, mask=mask, other=float("inf"))
    tl.store(out_ptr + offs, tl.sort(keys), mask=mask)


def block_sort(x, block=BLOCK):
    """Sorts each run of `block` consecutive keys on its own. One launch, no merge."""
    out = torch.empty_like(x)
    block_sort_kernel[(triton.cdiv(x.numel(), block),)](x, out, x.numel(), BLOCK=block)
    return out


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    x = torch.rand(n, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    print(f"{n} keys, blocks of {BLOCK}")
    for _ in range(2):
        # The first call compiles the kernel, so the second one is timed.
        sync()
        t0 = time.perf_counter()
        out = block_sort(x)
        sync()
        seconds = time.perf_counter() - t0
    ref = torch.cat([c.sort().values for c in x.split(BLOCK)])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    print(f"  block_sort {seconds * 1e3:9.3f} ms  {n / seconds / 1e6:8.1f} Mkeys/s")
    print("OK")
