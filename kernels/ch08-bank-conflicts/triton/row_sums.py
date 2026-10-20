# Triton has no __shared__ arrays. The compiler decides when a block goes
# through shared memory and picks a swizzled layout for it there.
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Rows per program and columns per step; tl.arange needs powers of two.
BLOCK_R = 32
BLOCK_C = 32


@triton.jit
def row_sums_kernel(a_ptr, out_ptr, rows, cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        mask = (r[:, None] < rows) & (c[None, :] < cols)
        tile = tl.load(a_ptr + r[:, None] * cols + c[None, :], mask=mask, other=0.0)
        acc += tl.sum(tile, axis=1)
    tl.store(out_ptr + r, acc, mask=r < rows)


def row_sums(a):
    rows, cols = a.shape
    out = torch.empty(rows, dtype=torch.float32, device=a.device)
    row_sums_kernel[(triton.cdiv(rows, BLOCK_R),)](a, out, rows, cols, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C)
    return out


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # Small integers keep every float sum exact, so any summation order matches.
    a = torch.randint(-8, 9, (4096, 256), generator=gen).float()
    torch.testing.assert_close(row_sums(a.to(DEVICE)).cpu(), a.sum(dim=1), rtol=0, atol=0)
    print("OK")
