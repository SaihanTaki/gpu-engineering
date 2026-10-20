import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy_calc import LIMITS, blocks_per_sm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 1024


@triton.jit
def saxpy_kernel(a, x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a * x + y, mask=mask)


def saxpy(a, x, y, num_warps=4, num_stages=3):
    """y = a * x + y in place. Returns the compiled kernel (None in the interpreter)."""
    n = y.numel()
    # num_warps sets the threads per program: 32 * num_warps. num_stages only
    # matters for loops, which this kernel does not have.
    return saxpy_kernel[(triton.cdiv(n, BLOCK),)](a, x, y, n, BLOCK=BLOCK, num_warps=num_warps,
                                                   num_stages=num_stages)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=gen).to(DEVICE)
    y0 = torch.rand(n, generator=gen).to(DEVICE)
    cc = torch.cuda.get_device_capability() if DEVICE == "cuda" else None

    for num_warps in (1, 2, 4, 8, 16):
        y = y0.clone()
        k = saxpy(2.0, x, y, num_warps=num_warps)
        torch.testing.assert_close(y, 2.0 * x + y0)
        if k is None:
            continue
        # Each program is one CUDA block of 32 * num_warps threads.
        threads = 32 * num_warps
        line = (f"num_warps={num_warps:2}: {threads:3} threads per program, {k.n_regs} registers per thread,"
                f" {k.n_spills} spilled, {k.metadata.shared} bytes shared memory")
        if cc in LIMITS:
            blocks, _ = blocks_per_sm(cc, k.n_regs, k.metadata.shared, threads)
            line += f", {blocks} programs per SM"
        print(line)
    if DEVICE == "cpu":
        print("Interpreter: no compiled kernel, so no register counts. Run on a GPU to see them.")
    print("OK")
