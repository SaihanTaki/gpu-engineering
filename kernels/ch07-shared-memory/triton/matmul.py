# Triton has no __shared__ and no barrier. A program loads whole tiles as
# block values; the compiler decides which of them to stage in shared memory
# and inserts the barriers itself. See generate.py for where it does that.
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# A power of two, as tl.arange requires. One 32 x 32 float32 tile is 4 KB.
BLOCK = 32


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    # One program computes one BLOCK x BLOCK tile of C, like one CUDA block.
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for p in range(0, tl.cdiv(n, BLOCK)):
        k = p * BLOCK + ks
        # One tile of A and one of B per phase, zeros past the edge.
        a = tl.load(a_ptr + rows[:, None] * n + k[None, :], mask=(rows[:, None] < n) & (k[None, :] < n), other=0.0)
        b = tl.load(b_ptr + k[:, None] * n + cols[None, :], mask=(k[:, None] < n) & (cols[None, :] < n), other=0.0)
        # "ieee" keeps full float32; the default, "tf32", may truncate the inputs to TF32.
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + rows[:, None] * n + cols[None, :], acc, mask=(rows[:, None] < n) & (cols[None, :] < n))


def matmul(a, b):
    n = a.shape[0]
    c = torch.empty((n, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(n, BLOCK), triton.cdiv(n, BLOCK))
    matmul_kernel[grid](a, b, c, n, BLOCK=BLOCK)
    return c


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    n = 256
    a = torch.rand((n, n), generator=gen) * 2 - 1
    b = torch.rand((n, n), generator=gen) * 2 - 1
    c = matmul(a.to(DEVICE), b.to(DEVICE)).cpu()
    torch.testing.assert_close(c, (a.double() @ b.double()).float(), rtol=1e-4, atol=1e-4 * n)
    print("OK")
