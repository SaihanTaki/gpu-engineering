import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def saxpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a * x + y, mask=mask)


def run(a, x, y, block=1024):
    x = torch.as_tensor(x, device=DEVICE)
    y = torch.as_tensor(y, device=DEVICE)
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    saxpy[grid](a, x, y, out, x.numel(), BLOCK=block)
    return out.cpu().numpy()


if __name__ == "__main__":
    x = torch.arange(8, dtype=torch.float32)
    y = torch.ones(8, dtype=torch.float32)
    print(run(2.0, x, y))
