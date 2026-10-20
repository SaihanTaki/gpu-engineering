import sys

import numpy as np
from numba import cuda, float32

# One warp per block; the block sums 32 rows, one row per lane.
TILE = 32


@cuda.jit
def row_sums_plain(a, out):
    # tile[lane, j] is word lane * 32 + j, so all 32 lanes of the warp hit bank j.
    tile = cuda.shared.array((TILE, TILE), float32)
    lane = cuda.threadIdx.x
    row0 = cuda.blockIdx.x * TILE
    rows, cols = a.shape
    acc = float32(0)
    for c0 in range(0, cols, TILE):
        # Lanes load 32 neighboring floats of one row: coalesced, and one tile row per store.
        for r in range(TILE):
            tile[r, lane] = a[row0 + r, c0 + lane] if row0 + r < rows else float32(0)
        cuda.syncthreads()
        # Each lane sums its own row, so at every j the warp reads column j of the tile.
        for j in range(TILE):
            acc += tile[lane, j]
        cuda.syncthreads()
    if row0 + lane < rows:
        out[row0 + lane] = acc


@cuda.jit
def row_sums_padded(a, out):
    # One extra column: tile[lane, j] is word lane * 33 + j, so lane k hits bank (k + j) % 32.
    tile = cuda.shared.array((TILE, TILE + 1), float32)
    lane = cuda.threadIdx.x
    row0 = cuda.blockIdx.x * TILE
    rows, cols = a.shape
    acc = float32(0)
    for c0 in range(0, cols, TILE):
        for r in range(TILE):
            tile[r, lane] = a[row0 + r, c0 + lane] if row0 + r < rows else float32(0)
        cuda.syncthreads()
        for j in range(TILE):
            acc += tile[lane, j]
        cuda.syncthreads()
    if row0 + lane < rows:
        out[row0 + lane] = acc


@cuda.jit
def row_sums_swizzled(a, out):
    # No padding: element (r, c) is stored in column c ^ r, which spreads every column over all banks.
    tile = cuda.shared.array((TILE, TILE), float32)
    lane = cuda.threadIdx.x
    row0 = cuda.blockIdx.x * TILE
    rows, cols = a.shape
    acc = float32(0)
    for c0 in range(0, cols, TILE):
        for r in range(TILE):
            tile[r, lane ^ r] = a[row0 + r, c0 + lane] if row0 + r < rows else float32(0)
        cuda.syncthreads()
        for j in range(TILE):
            acc += tile[lane, j ^ lane]
        cuda.syncthreads()
    if row0 + lane < rows:
        out[row0 + lane] = acc


def run(kernel, a):
    rows, cols = a.shape
    if cols % TILE:
        raise ValueError("the column count must be a multiple of 32")
    d_a = cuda.to_device(a)
    d_out = cuda.device_array(rows, dtype=np.float32)
    kernel[(rows + TILE - 1) // TILE, TILE](d_a, d_out)
    return d_out.copy_to_host()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    # Small integers keep every float sum exact, so any summation order matches.
    a = np.random.default_rng(0).integers(-8, 9, size=(rows, 256)).astype(np.float32)
    expected = a.sum(axis=1)
    for k in (row_sums_plain, row_sums_padded, row_sums_swizzled):
        assert np.array_equal(run(k, a), expected)
    print("OK")
