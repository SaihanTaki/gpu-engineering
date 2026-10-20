import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# One program per BLOCK_H x BLOCK_W output tile; tl.arange needs powers of two.
BLOCK_H = 16
BLOCK_W = 32


@triton.jit
def conv2d_kernel(in_ptr, f_ptr, out_ptr, h, w, R: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    cols = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    # static_range unrolls at compile time: (2R+1)^2 shifted loads of the tile.
    for i in tl.static_range(2 * R + 1):
        for j in tl.static_range(2 * R + 1):
            r = rows + (i - R)
            c = cols + (j - R)
            # Masked-off pixels read as other=0.0: that is the zero padding.
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            pixels = tl.load(in_ptr + r * w + c, mask=inside, other=0.0)
            acc += tl.load(f_ptr + i * (2 * R + 1) + j) * pixels
    tl.store(out_ptr + rows * w + cols, acc, mask=(rows < h) & (cols < w))


def conv2d(img, filt):
    h, w = img.shape
    R = filt.shape[0] // 2
    out = torch.empty_like(img)
    grid = (triton.cdiv(h, BLOCK_H), triton.cdiv(w, BLOCK_W))
    conv2d_kernel[grid](img, filt.contiguous(), out, h, w, R=R, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W)
    return out


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    w = int(sys.argv[2]) if len(sys.argv) > 2 else h
    reps = 20
    gen = torch.Generator().manual_seed(0)
    img = (torch.rand((h, w), generator=gen) * 2 - 1).to(DEVICE)
    nbytes = img.numel() * img.element_size()

    # Effective bandwidth counts each pixel read once and written once.
    print(f"{h} x {w} floats, BLOCK = {BLOCK_H} x {BLOCK_W}")
    for R in (1, 2, 3):
        filt = (torch.rand((2 * R + 1, 2 * R + 1), generator=gen) * 2 - 1).to(DEVICE)
        # The first call compiles the kernel, so it is not timed.
        conv2d(img, filt)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            conv2d(img, filt)
        sync()
        seconds = (time.perf_counter() - t0) / reps
        print(f"  R={R} {seconds * 1e3:9.3f} ms  {2 * nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
