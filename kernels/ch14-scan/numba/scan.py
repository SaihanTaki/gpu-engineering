import sys
import time

import numpy as np
from numba import cuda, int32

BLOCK = 256
# Blelloch gives each thread two elements, so a block scans 512 of them.
ITEMS = 2 * BLOCK


@cuda.jit
def scan_hillis_steele(inp, out, block_sums, inclusive):
    s = cuda.shared.array(BLOCK, int32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * BLOCK + tid
    s[tid] = inp[i] if i < inp.size else 0
    cuda.syncthreads()

    offset = 1
    while offset < BLOCK:
        # Read every left neighbour before anyone overwrites it.
        left = s[tid - offset] if tid >= offset else 0
        cuda.syncthreads()
        if tid >= offset:
            s[tid] += left
        cuda.syncthreads()
        offset *= 2

    if tid == BLOCK - 1:
        block_sums[cuda.blockIdx.x] = s[BLOCK - 1]
    if i < inp.size:
        # The exclusive scan is the inclusive one shifted right by one.
        if inclusive:
            out[i] = s[tid]
        else:
            out[i] = s[tid - 1] if tid > 0 else 0


@cuda.jit
def scan_blelloch(inp, out, block_sums, inclusive):
    s = cuda.shared.array(ITEMS, int32)
    total = cuda.shared.array(1, int32)
    tid = cuda.threadIdx.x
    base = cuda.blockIdx.x * ITEMS
    n = inp.size
    # Two loads per thread, BLOCK apart, so each warp reads consecutive words.
    for k in range(tid, ITEMS, BLOCK):
        s[k] = inp[base + k] if base + k < n else 0

    offset = 1
    d = ITEMS // 2
    while d > 0:
        cuda.syncthreads()
        if tid < d:
            a = offset * (2 * tid + 1) - 1
            b = offset * (2 * tid + 2) - 1
            s[b] += s[a]
        offset *= 2
        d //= 2

    if tid == 0:
        total[0] = s[ITEMS - 1]
        block_sums[cuda.blockIdx.x] = total[0]
        s[ITEMS - 1] = 0

    d = 1
    while d < ITEMS:
        offset //= 2
        cuda.syncthreads()
        if tid < d:
            a = offset * (2 * tid + 1) - 1
            b = offset * (2 * tid + 2) - 1
            t = s[a]
            s[a] = s[b]
            s[b] += t
        d *= 2
    cuda.syncthreads()

    for k in range(tid, ITEMS, BLOCK):
        if base + k < n:
            # s is the exclusive scan; the inclusive one is s shifted left, ending in the total.
            if not inclusive:
                out[base + k] = s[k]
            elif k + 1 < ITEMS:
                out[base + k] = s[k + 1]
            else:
                out[base + k] = total[0]


@cuda.jit
def add_offsets(out, offsets, per_block):
    i = cuda.grid(1)
    if i < out.size:
        out[i] += offsets[i // per_block]


ALGOS = {"hillis-steele": (scan_hillis_steele, BLOCK), "blelloch": (scan_blelloch, ITEMS)}


def scratch_for(n):
    # 2 ints per tile at every level, as in scan.cu.
    return cuda.device_array(n // 64 + 64, dtype=np.int32)


def scan_device(algo, inclusive, d_in, d_out, scratch=None):
    """Scan-then-add: scan every tile, scan the tile totals, add each tile's offset."""
    kernel, per_block = ALGOS[algo]
    n = d_in.size
    tiles = (n + per_block - 1) // per_block
    if scratch is None:
        scratch = scratch_for(n)
    sums = scratch[:tiles]
    kernel[tiles, BLOCK](d_in, d_out, sums, inclusive)
    if tiles == 1:
        return
    offsets = scratch[tiles : 2 * tiles]
    # The same scan, exclusive, on the tile totals; recursion handles more than one tile of them.
    scan_device(algo, False, sums, offsets, scratch[2 * tiles :])
    add_offsets[(n + 255) // 256, 256](d_out, offsets, per_block)


def scan(x, algo="blelloch", inclusive=True):
    d_in = cuda.to_device(np.ascontiguousarray(x, dtype=np.int32))
    d_out = cuda.device_array(x.size, dtype=np.int32)
    scan_device(algo, inclusive, d_in, d_out)
    return d_out.copy_to_host()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    x = np.random.default_rng(0).integers(0, 10, n, dtype=np.int32)
    d_in = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.int32)
    scratch = scratch_for(n)
    incl = np.cumsum(x, dtype=np.int64).astype(np.int32)

    # Effective bandwidth counts each element read once and written once.
    print(f"n = {n} ints, {BLOCK} threads per block")
    for algo in ALGOS:
        for inclusive in (True, False):
            # The first call compiles the kernels and is checked, not timed.
            scan_device(algo, inclusive, d_in, d_out, scratch)
            want = incl if inclusive else incl - x
            np.testing.assert_array_equal(d_out.copy_to_host(), want)
            cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                scan_device(algo, inclusive, d_in, d_out, scratch)
            cuda.synchronize()
            seconds = (time.perf_counter() - t0) / reps
            kind = "inclusive" if inclusive else "exclusive"
            print(f"  {algo:<14} {kind:<9} {seconds * 1e3:9.3f} ms  {2 * x.nbytes / seconds / 1e9:8.1f} GB/s")
    print("OK")
