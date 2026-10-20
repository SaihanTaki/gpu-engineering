# A Triton program works on a whole block of elements at once, so there is
# no threadIdx: program_id plays the role of blockIdx, tl.arange covers the rest.
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def whoami(out_ptr, block_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # The last program can cover offsets past the end of the array.
    mask = offs < n
    tl.store(out_ptr + offs, offs, mask=mask)
    tl.store(block_ptr + offs, tl.zeros_like(offs) + pid, mask=mask)


@triton.jit
def whoami2d(out_ptr, width, height, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    rows = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)
    # Broadcasting a column of rows against a row of cols gives a 2D tile.
    idx = rows[:, None] * width + cols[None, :]
    mask = (rows[:, None] < height) & (cols[None, :] < width)
    tl.store(out_ptr + idx, idx, mask=mask)


def run(n, block):
    # tl.arange only accepts power-of-two sizes, so BLOCK must be one.
    out = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    blocks = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    grid = (triton.cdiv(n, block),)
    whoami[grid](out, blocks, n, BLOCK=block)
    return out.cpu().numpy(), blocks.cpu().numpy()


def run2d(width, height, block=(2, 4)):
    out = torch.full((height, width), -1, dtype=torch.int32, device=DEVICE)
    grid = (triton.cdiv(width, block[0]), triton.cdiv(height, block[1]))
    whoami2d[grid](out, width, height, BLOCK_X=block[0], BLOCK_Y=block[1])
    return out.cpu().numpy()


if __name__ == "__main__":
    out, blocks = run(20, 8)
    print(f"{'i':>5} {'block':>6}")
    for i, b in zip(out, blocks):
        print(f"{i:>5} {b:>6}")
