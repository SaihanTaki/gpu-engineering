# A Triton program works on a whole block of elements, so there is no
# per-lane if/else. A per-element choice is written with tl.where.
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tl.arange only accepts power-of-two sizes, so BLOCK must be one.
BLOCK = 1024


@triton.jit
def select_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Both results are computed for every element; tl.where keeps one per element.
    out = tl.where(x > 0, x * 2.0 + 1.0, x * 0.5)
    tl.store(out_ptr + offs, out, mask=mask)


def select(x):
    out = torch.empty_like(x)
    n = out.numel()
    select_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=gen)
    out = select(x.to(DEVICE)).cpu()
    torch.testing.assert_close(out, torch.where(x > 0, x * 2 + 1, x * 0.5), rtol=0, atol=0)
    print("OK")
