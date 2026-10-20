import sys

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 256
# Elements of known value after each buffer, so a stray read or write shows up.
GUARD = BLOCK
SENTINEL = -7.0


@triton.jit
def scale_unmasked_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, SHOW: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if SHOW:
        tl.device_print("offs", offs)
    # No mask: the last program loads and stores BLOCK elements even when fewer are left.
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, 2 * x)


@triton.jit
def scale_masked_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, SHOW: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if SHOW:
        tl.device_print("mask", mask)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, 2 * x, mask=mask)


KERNELS = {"unmasked": scale_unmasked_kernel, "masked": scale_masked_kernel}


def guarded(n, fill=0.0):
    # One allocation with a guard band after the first n elements. Out-of-bounds
    # accesses then land in memory we own and can inspect, not in someone else's.
    buf = torch.full((n + GUARD,), SENTINEL, dtype=torch.float32, device=DEVICE)
    buf[:n] = fill
    return buf


def scale(variant, x, show=False):
    """Returns (2 * x as computed, True if both guard bands are untouched)."""
    n = x.numel()
    xb = guarded(n)
    xb[:n] = x.to(DEVICE)
    ob = guarded(n)
    KERNELS[variant][(triton.cdiv(n, BLOCK),)](xb, ob, n, BLOCK=BLOCK, SHOW=show)
    intact = bool((ob[n:] == SENTINEL).all()) and bool((xb[n:] == SENTINEL).all())
    return ob[:n].cpu(), intact


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    x = torch.arange(n, dtype=torch.float32)
    print(f"n = {n}, BLOCK = {BLOCK}, {triton.cdiv(n, BLOCK)} programs")
    for variant in KERNELS:
        out, intact = scale(variant, x)
        ok = torch.equal(out, 2 * x)
        print(f"  {variant:<8} result {'OK' if ok else 'WRONG'}, guard band {'intact' if intact else 'OVERWRITTEN'}")
