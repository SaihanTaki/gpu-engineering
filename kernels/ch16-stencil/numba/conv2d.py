import sys
import time

import numpy as np
from numba import cuda, float32

# Blocks of TILE x TILE threads, one thread per output pixel.
TILE = 16


def make_kernels(filt):
    """The naive and the tiled kernel for one (2R+1) x (2R+1) filter.

    cuda.const.array_like copies the filter into constant memory when the
    kernel compiles, so each filter gets its own pair of kernels. Pixels
    outside the image count as zero.
    """
    filt = np.ascontiguousarray(filt, dtype=np.float32)
    R = filt.shape[0] // 2
    # The shared array shape must be a compile-time constant.
    IN = TILE + 2 * R

    @cuda.jit
    def conv2d_naive(img, out):
        f = cuda.const.array_like(filt)
        h, w = img.shape
        x, y = cuda.grid(2)
        if x >= w or y >= h:
            return
        acc = float32(0.0)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    acc += f[dy + R, dx + R] * img[yy, xx]
        out[y, x] = acc

    @cuda.jit
    def conv2d_tiled(img, out):
        f = cuda.const.array_like(filt)
        tile = cuda.shared.array((IN, IN), float32)
        h, w = img.shape
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        # Image coordinates of tile[0, 0], the top-left corner of the halo.
        x0 = cuda.blockIdx.x * TILE - R
        y0 = cuda.blockIdx.y * TILE - R
        # IN * IN slots and only TILE * TILE threads, so each thread fills several.
        for i in range(ty * TILE + tx, IN * IN, TILE * TILE):
            yy = y0 + i // IN
            xx = x0 + i % IN
            if 0 <= yy < h and 0 <= xx < w:
                tile[i // IN, i % IN] = img[yy, xx]
            else:
                tile[i // IN, i % IN] = float32(0.0)
        cuda.syncthreads()

        x = x0 + R + tx
        y = y0 + R + ty
        # Past the only barrier, so threads outside the image may stop here.
        if x >= w or y >= h:
            return
        acc = float32(0.0)
        for dy in range(2 * R + 1):
            for dx in range(2 * R + 1):
                acc += f[dy, dx] * tile[ty + dy, tx + dx]
        out[y, x] = acc

    return {"naive": conv2d_naive, "tiled": conv2d_tiled}


def launch(kernel, d_img, d_out):
    h, w = d_img.shape
    grid = ((w + TILE - 1) // TILE, (h + TILE - 1) // TILE)
    kernel[grid, (TILE, TILE)](d_img, d_out)


def run(name, img, filt):
    d_img = cuda.to_device(np.ascontiguousarray(img, dtype=np.float32))
    d_out = cuda.device_array(img.shape, dtype=np.float32)
    launch(make_kernels(filt)[name], d_img, d_out)
    return d_out.copy_to_host()


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    w = int(sys.argv[2]) if len(sys.argv) > 2 else h
    reps = 20
    rng = np.random.default_rng(0)
    img = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    d_img = cuda.to_device(img)
    d_out = cuda.device_array((h, w), dtype=np.float32)

    # Effective bandwidth counts each pixel read once and written once.
    print(f"{h} x {w} floats, {TILE} x {TILE} threads per block")
    for R in (1, 2, 3):
        filt = rng.uniform(-1, 1, (2 * R + 1, 2 * R + 1)).astype(np.float32)
        for name, kernel in make_kernels(filt).items():
            # The first call compiles the kernel, so it is not timed.
            launch(kernel, d_img, d_out)
            cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                launch(kernel, d_img, d_out)
            cuda.synchronize()
            seconds = (time.perf_counter() - t0) / reps
            print(f"  R={R} {name:<6} {seconds * 1e3:9.3f} ms  {2 * img.nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
