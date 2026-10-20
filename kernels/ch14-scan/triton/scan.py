import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Elements per program; tl.arange needs a power of two.
BLOCK = 1024


@triton.jit
def add(a, b):
    return a + b


@triton.jit
def scan_block_kernel(x_ptr, out_ptr, sums_ptr, n, BLOCK: tl.constexpr, INCLUSIVE: tl.constexpr, USE_CUMSUM: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if USE_CUMSUM:
        s = tl.cumsum(x, 0)
    else:
        # Any associative combine function works here; tl.cumsum is this one with add.
        s = tl.associative_scan(x, 0, add)
    tl.store(sums_ptr + pid, tl.sum(x, 0))
    if INCLUSIVE:
        out = s
    else:
        # Subtracting works for integer addition; other operators need a shift instead.
        out = s - x
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def add_offsets_kernel(out_ptr, offsets_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    out = tl.load(out_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, out + tl.load(offsets_ptr + pid), mask=mask)


VARIANTS = {"associative_scan": False, "cumsum": True}


def scan(x, inclusive=True, variant="associative_scan", block=BLOCK):
    """Scan-then-add: scan every tile, scan the tile totals, add each tile's offset."""
    n = x.numel()
    tiles = triton.cdiv(n, block)
    out = torch.empty_like(x)
    sums = torch.empty(tiles, dtype=x.dtype, device=x.device)
    scan_block_kernel[(tiles,)](x, out, sums, n, BLOCK=block, INCLUSIVE=inclusive, USE_CUMSUM=VARIANTS[variant])
    if tiles > 1:
        # The same scan, exclusive, on the tile totals; recursion handles more than one tile of them.
        offsets = scan(sums, inclusive=False, variant=variant, block=block)
        add_offsets_kernel[(tiles,)](out, offsets, n, BLOCK=block)
    return out


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    x = torch.randint(0, 10, (n,), dtype=torch.int32, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    incl = torch.cumsum(x, 0, dtype=torch.int32)
    nbytes = x.numel() * x.element_size()

    # Effective bandwidth counts each element read once and written once.
    print(f"n = {n} ints, BLOCK = {BLOCK}")
    for variant in VARIANTS:
        for inclusive in (True, False):
            # The first call compiles the kernels and is checked, not timed.
            out = scan(x, inclusive, variant)
            torch.testing.assert_close(out, incl if inclusive else incl - x, rtol=0, atol=0)
            sync()
            t0 = time.perf_counter()
            for _ in range(reps):
                out = scan(x, inclusive, variant)
            sync()
            seconds = (time.perf_counter() - t0) / reps
            kind = "inclusive" if inclusive else "exclusive"
            print(f"  {variant:<16} {kind:<9} {seconds * 1e3:9.3f} ms  {2 * nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
