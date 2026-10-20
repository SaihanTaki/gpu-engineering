import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 1024
TABLE = 256


@triton.jit
def scale_by_table(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr, TABLE: tl.constexpr,
                   X_CACHE: tl.constexpr, X_EVICT: tl.constexpr, W_EVICT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # x is read once, so it should not push reusable data out of the caches.
    x = tl.load(x_ptr + offs, mask=mask, cache_modifier=X_CACHE, eviction_policy=X_EVICT)
    # Every program reads the same small table w.
    w = tl.load(w_ptr + offs % TABLE, mask=mask, eviction_policy=W_EVICT)
    tl.store(out_ptr + offs, x * w, mask=mask)


# The variants generate.py compiles to PTX: (X_CACHE, X_EVICT, W_EVICT).
VARIANTS = {
    "default": ("", "", ""),
    "stream x, keep w": ("", "evict_first", "evict_last"),
    "x in L2 only": (".cg", "", ""),
    "x streaming": (".cs", "", ""),
}


def run(x, w, variant="default"):
    x_cache, x_evict, w_evict = VARIANTS[variant]
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK),)
    scale_by_table[grid](x, w, out, x.numel(), BLOCK=BLOCK, TABLE=TABLE,
                         X_CACHE=x_cache, X_EVICT=x_evict, W_EVICT=w_evict)
    return out


if __name__ == "__main__":
    x = torch.randn(1 << 20, device=DEVICE)
    w = torch.randn(TABLE, device=DEVICE)
    expected = x * w[torch.arange(x.numel(), device=DEVICE) % TABLE]
    for name in VARIANTS:
        torch.testing.assert_close(run(x, w, name), expected)
    print("OK")
