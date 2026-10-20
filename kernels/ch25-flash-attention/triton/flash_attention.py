import math
import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, n,
    HEAD_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
):
    # One program per (Q block, batch * head): FlashAttention-2's outer loop runs in parallel.
    start_m = tl.program_id(0) * BLOCK_M
    head = tl.program_id(1).to(tl.int64) * n * HEAD_D
    rows = start_m + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_D)
    row_ok = rows < n

    q = tl.load(q_ptr + head + rows[:, None] * HEAD_D + dims[None, :], mask=row_ok[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_D], tl.float32)

    # With a causal mask, K/V blocks that start after this block's last row are all masked: skip them.
    hi = n
    if CAUSAL:
        hi = tl.minimum(n, start_m + BLOCK_M)
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        col_ok = cols < n
        k = tl.load(k_ptr + head + cols[:, None] * HEAD_D + dims[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_ptr + head + cols[:, None] * HEAD_D + dims[None, :], mask=col_ok[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (cols[None, :] <= rows[:, None])
        s = tl.where(keep, s, float("-inf"))

        # Online softmax: a larger max shrinks everything summed so far by exp(m_old - m_new).
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # FlashAttention-2 divides by the row sum once, after the last block.
    acc = acc / l_i[:, None]
    tl.store(o_ptr + head + rows[:, None] * HEAD_D + dims[None, :], acc.to(o_ptr.dtype.element_ty), mask=row_ok[:, None])


def flash_attention(q, k, v, causal=False, block_m=64, block_n=64):
    """q, k, v: (batch, heads, n, d), contiguous, d a power of two >= 16."""
    b, h, n, d = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(n, block_m), b * h)
    flash_fwd_kernel[grid](q, k, v, o, 1.0 / math.sqrt(d), n, HEAD_D=d, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal)
    return o


def naive_attention(q, k, v, causal=False):
    # Materializes the n x n matrices S and P in GPU memory, as Algorithm 0 of Dao et al. 2022 does.
    n, d = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    if causal:
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return p @ v


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, reps):
    fn()  # compiles on the first call
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps * 1e3


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    b, h, d = 4, 16, 64
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    reps = 10 if DEVICE == "cuda" else 1
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, generator=g).to(DEVICE, dtype) for _ in range(3))

    print(f"batch {b}, heads {h}, n {n}, d {d}, {dtype}")
    for causal in (False, True):
        want = naive_attention(q.float(), k.float(), v.float(), causal)
        got = flash_attention(q, k, v, causal)
        torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)
        flops = 4 * b * h * n * n * d / (2 if causal else 1)
        for name, fn in [
            ("naive", lambda: naive_attention(q, k, v, causal)),
            ("flash (this file)", lambda: flash_attention(q, k, v, causal)),
            ("torch sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)),
        ]:
            ms = time_ms(fn, reps)
            print(f"  causal={causal!s:<5} {name:<18} {ms:9.3f} ms  {flops / ms / 1e9:8.1f} TFLOPS")
    print("OK")
