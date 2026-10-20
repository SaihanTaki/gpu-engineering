import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# One program per BLOCK x BLOCK tile; tl.arange needs a power of two.
BLOCK = 32


@triton.jit
def transpose_trans_kernel(in_ptr, out_ptr, rows, cols, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    tile = tl.load(in_ptr + r[:, None] * cols + c[None, :], mask=mask)
    # Rows of out are columns of in, so out's tile is indexed [c, r].
    out_offs = c[:, None] * rows + r[None, :]
    out_mask = (c[:, None] < cols) & (r[None, :] < rows)
    tl.store(out_ptr + out_offs, tl.trans(tile), mask=out_mask)


@triton.jit
def transpose_strides_kernel(in_ptr, out_ptr, rows, cols, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    tile = tl.load(in_ptr + r[:, None] * cols + c[None, :], mask=mask)
    # Same tile, same shape: only the output strides are swapped, like the naive CUDA kernel.
    tl.store(out_ptr + c[None, :] * rows + r[:, None], tile, mask=mask)


KERNELS = {"trans": transpose_trans_kernel, "strides": transpose_strides_kernel}


def transpose(a, variant="trans"):
    rows, cols = a.shape
    out = torch.empty((cols, rows), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(rows, BLOCK), triton.cdiv(cols, BLOCK))
    KERNELS[variant][grid](a, out, rows, cols, BLOCK=BLOCK)
    return out


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else rows
    reps = 20
    a = torch.rand((rows, cols), generator=torch.Generator().manual_seed(0)).to(DEVICE)
    nbytes = a.numel() * a.element_size()

    # Effective bandwidth counts each element read once and written once.
    print(f"{rows} x {cols} floats, BLOCK = {BLOCK}")
    for variant in KERNELS:
        # The first call compiles the kernel, so it is not timed.
        out = transpose(a, variant)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = transpose(a, variant)
        sync()
        seconds = (time.perf_counter() - t0) / reps
        torch.testing.assert_close(out, a.T, rtol=0, atol=0)
        print(f"  {variant:<8} {seconds * 1e3:9.3f} ms  {2 * nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
