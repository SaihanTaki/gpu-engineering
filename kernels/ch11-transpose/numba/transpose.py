import sys
import time

import numpy as np
from numba import cuda, float32

# Each block of TILE x BLOCK_ROWS threads handles one TILE x TILE tile.
TILE = 32
BLOCK_ROWS = 8


@cuda.jit
def transpose_naive(inp, out):
    rows, cols = inp.shape
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    for j in range(0, TILE, BLOCK_ROWS):
        if x < cols and y + j < rows:
            # The read is coalesced; neighbouring lanes write a column of out.
            out[x, y + j] = inp[y + j, x]


def make_shared(pad):
    # The shared array shape must be a compile-time constant, so each pad gets its own kernel.
    shape = (TILE, TILE + pad)

    @cuda.jit
    def transpose_shared(inp, out):
        tile = cuda.shared.array(shape, float32)
        rows, cols = inp.shape
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        x = cuda.blockIdx.x * TILE + tx
        y = cuda.blockIdx.y * TILE + ty
        for j in range(0, TILE, BLOCK_ROWS):
            if x < cols and y + j < rows:
                tile[ty + j, tx] = inp[y + j, x]

        # Each thread now reads elements that other warps wrote.
        cuda.syncthreads()

        # Swap the block indices: this block writes the mirrored tile of out.
        x = cuda.blockIdx.y * TILE + tx
        y = cuda.blockIdx.x * TILE + ty
        for j in range(0, TILE, BLOCK_ROWS):
            if x < rows and y + j < cols:
                out[y + j, x] = tile[tx, ty + j]

    return transpose_shared


transpose_tiled = make_shared(0)
transpose_padded = make_shared(1)

KERNELS = {"naive": transpose_naive, "tiled": transpose_tiled, "padded": transpose_padded}


def launch(kernel, d_in, d_out):
    rows, cols = d_in.shape
    grid = ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE)
    kernel[grid, (TILE, BLOCK_ROWS)](d_in, d_out)


def run(name, a):
    d_in = cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    d_out = cuda.device_array((a.shape[1], a.shape[0]), dtype=np.float32)
    launch(KERNELS[name], d_in, d_out)
    return d_out.copy_to_host()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else rows
    reps = 20
    a = np.random.default_rng(0).random((rows, cols), dtype=np.float32)
    d_in = cuda.to_device(a)
    d_out = cuda.device_array((cols, rows), dtype=np.float32)

    # Effective bandwidth counts each element read once and written once.
    print(f"{rows} x {cols} floats, {TILE} x {BLOCK_ROWS} threads per block")
    for name, kernel in KERNELS.items():
        # The first call compiles the kernel, so it is not timed.
        launch(kernel, d_in, d_out)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            launch(kernel, d_in, d_out)
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        np.testing.assert_array_equal(d_out.copy_to_host(), a.T)
        print(f"  {name:<7} {seconds * 1e3:9.3f} ms  {2 * a.nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
