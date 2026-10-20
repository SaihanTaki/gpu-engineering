# Triton picks which thread loads which element, but the addresses come from
# your offsets. A strided offset vector is a strided load, whoever issues it.
import warnings

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tl.arange only accepts power-of-two sizes, so BLOCK must be one.
BLOCK = 1024


@triton.jit
def copy_strided_kernel(src_ptr, out_ptr, n, stride, offset, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # stride 1 gives contiguous offsets; any other stride spreads the block over memory.
    x = tl.load(src_ptr + offs * stride + offset, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def get_x_aos_kernel(p_ptr, out_ptr, n, FIELDS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # The AoS array as an n x FIELDS matrix; a BLOCK x 1 block is one field of BLOCK structs.
    src = tl.make_block_ptr(
        p_ptr, shape=(n, FIELDS), strides=(FIELDS, 1), offsets=(pid * BLOCK, 0), block_shape=(BLOCK, 1), order=(1, 0)
    )
    x = tl.load(src, boundary_check=(0,))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.reshape(x, (BLOCK,)), mask=offs < n)


def copy_strided(src, n, stride=1, offset=0):
    out = torch.empty(n, dtype=src.dtype, device=src.device)
    copy_strided_kernel[(triton.cdiv(n, BLOCK),)](src, out, n, stride, offset, BLOCK=BLOCK)
    return out


def get_x_aos(p):
    n, fields = p.shape
    out = torch.empty(n, dtype=p.dtype, device=p.device)
    with warnings.catch_warnings():
        # Triton 3.x marks make_block_ptr deprecated in favour of tensor descriptors; it still works.
        warnings.simplefilter("ignore", UserWarning)
        get_x_aos_kernel[(triton.cdiv(n, BLOCK),)](p, out, n, FIELDS=fields, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    n = 1 << 16
    src = torch.arange(n * 32, dtype=torch.float32, device=DEVICE)
    for stride, offset in [(1, 0), (1, 1), (2, 0), (32, 0)]:
        expected = src[offset : offset + n * stride : stride]
        torch.testing.assert_close(copy_strided(src, n, stride, offset), expected, rtol=0, atol=0)
    p = torch.randn(n, 3, device=DEVICE)
    torch.testing.assert_close(get_x_aos(p), p[:, 0], rtol=0, atol=0)
    print("OK")
