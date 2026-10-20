import sys
import time

import numpy as np
from numba import cuda, float32

THREADS = 256  # threads per block for bitonic_step
TILE = 1024  # keys per block in the shared memory kernel


@cuda.jit
def bitonic_step(a, j, k):
    t = cuda.grid(1)
    if t >= a.size // 2:
        return
    # Lower key of comparator t in a stage of distance j: skip the upper half of every 2j group.
    i = (t // j) * 2 * j + t % j
    p = i + j
    up = (i & k) == 0
    if (a[i] > a[p]) == up:
        a[i], a[p] = a[p], a[i]


def make_shared(tile):
    # The shared array size must be a compile-time constant, so each tile size gets its own kernel.
    half = tile // 2

    @cuda.jit
    def bitonic_shared(a, k_first, k_last):
        s = cuda.shared.array(tile, float32)
        base = cuda.blockIdx.x * tile
        t = cuda.threadIdx.x
        s[t] = a[base + t]
        s[t + half] = a[base + t + half]
        cuda.syncthreads()
        k = k_first
        while k <= k_last:
            j = min(k, tile) // 2
            while j > 0:
                i = (t // j) * 2 * j + t % j
                p = i + j
                # The direction depends on the global index, so neighbouring tiles sort opposite ways.
                up = ((base + i) & k) == 0
                if (s[i] > s[p]) == up:
                    s[i], s[p] = s[p], s[i]
                cuda.syncthreads()
                j //= 2
            k *= 2
        a[base + t] = s[t]
        a[base + t + half] = s[t + half]

    return bitonic_shared


bitonic_shared = make_shared(TILE)
_shared = {TILE: bitonic_shared}


def sort_global(d, m, tile):
    # Every stage is its own launch: the launch boundary is the barrier between stages.
    blocks = (m // 2 + THREADS - 1) // THREADS
    k = 2
    while k <= m:
        j = k // 2
        while j > 0:
            bitonic_step[blocks, THREADS](d, j, k)
            j //= 2
        k *= 2


def sort_shared(d, m, tile):
    # Stages with j < tile run inside shared memory; only j >= tile needs a global launch.
    if tile not in _shared:
        _shared[tile] = make_shared(tile)
    kernel = _shared[tile]
    tiles = m // tile
    blocks = (m // 2 + THREADS - 1) // THREADS
    kernel[tiles, tile // 2](d, 2, tile)
    k = 2 * tile
    while k <= m:
        j = k // 2
        while j >= tile:
            bitonic_step[blocks, THREADS](d, j, k)
            j //= 2
        kernel[tiles, tile // 2](d, k, k)
        k *= 2


METHODS = {"global": sort_global, "shared": sort_shared}


def padded_size(n, tile):
    m = tile
    while m < n:
        m *= 2
    return m


def sort(x, method="shared", tile=TILE):
    x = np.asarray(x, dtype=np.float32)
    m = padded_size(x.size, tile)
    # Pad to a power of two with +inf, which sorts to the end.
    padded = np.full(m, np.inf, dtype=np.float32)
    padded[: x.size] = x
    d = cuda.to_device(padded)
    METHODS[method](d, m, tile)
    return d.copy_to_host()[: x.size]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    x = np.random.default_rng(0).uniform(-1, 1, n).astype(np.float32)
    m = padded_size(n, TILE)
    padded = np.full(m, np.inf, dtype=np.float32)
    padded[:n] = x
    print(f"{n} keys, padded to {m}")
    for name, method in METHODS.items():
        # The first run compiles the kernels, so time a second run on fresh input.
        for _ in range(2):
            d = cuda.to_device(padded)
            cuda.synchronize()
            t0 = time.perf_counter()
            method(d, m, TILE)
            cuda.synchronize()
            seconds = time.perf_counter() - t0
        np.testing.assert_array_equal(d.copy_to_host()[:n], np.sort(x))
        print(f"  {name:<7} {seconds * 1e3:9.3f} ms  {n / seconds / 1e6:8.1f} Mkeys/s")
    print("OK")
