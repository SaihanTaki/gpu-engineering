import sys
import time

import numpy as np
from numba import cuda, uint32

BINS = 256
THREADS = 256


@cuda.jit
def hist_global(data, hist):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, data.size, stride):
        # Every element is one atomic on global memory.
        cuda.atomic.add(hist, data[i], 1)


@cuda.jit
def hist_shared(data, hist):
    local = cuda.shared.array(BINS, uint32)
    tx = cuda.threadIdx.x
    for b in range(tx, BINS, cuda.blockDim.x):
        local[b] = 0
    cuda.syncthreads()

    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, data.size, stride):
        # Only threads of this block compete for this copy.
        cuda.atomic.add(local, data[i], 1)
    # Every count must be final before it is merged.
    cuda.syncthreads()

    for b in range(tx, BINS, cuda.blockDim.x):
        if local[b] > 0:
            cuda.atomic.add(hist, b, local[b])


KERNELS = {"global": hist_global, "shared": hist_shared}


def launch(kernel, d_data, d_hist, blocks, threads=THREADS):
    kernel[blocks, threads](d_data, d_hist)


def run(name, data, blocks=4, threads=THREADS):
    d_data = cuda.to_device(np.ascontiguousarray(data, dtype=np.uint8))
    d_hist = cuda.to_device(np.zeros(BINS, dtype=np.uint32))
    launch(KERNELS[name], d_data, d_hist, blocks, threads)
    return d_hist.copy_to_host()


def make_input(dist, n, seed=0):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        return rng.integers(0, BINS, n, dtype=np.uint8)
    if dist == "skewed":
        # Bin b has probability 2^-(b+1), capped at the last bin.
        return np.minimum(rng.geometric(0.5, n) - 1, BINS - 1).astype(np.uint8)
    return np.full(n, 7, dtype=np.uint8)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    reps = 20
    # The simulator's fake device has no SM count; one SM's worth of blocks is enough there.
    blocks = getattr(cuda.get_current_device(), "MULTIPROCESSOR_COUNT", 1) * 8
    d_hist = cuda.device_array(BINS, dtype=np.uint32)
    print(f"{n} bytes, {blocks} blocks of {THREADS} threads")
    for dist in ("uniform", "skewed", "one-bin"):
        data = make_input(dist, n)
        d_data = cuda.to_device(data)
        ref = np.bincount(data, minlength=BINS)
        for name, kernel in KERNELS.items():
            # The first call compiles the kernel, so it is not timed.
            d_hist.copy_to_device(np.zeros(BINS, dtype=np.uint32))
            launch(kernel, d_data, d_hist, blocks)
            cuda.synchronize()
            np.testing.assert_array_equal(d_hist.copy_to_host(), ref)
            seconds = 0.0
            for _ in range(reps):
                d_hist.copy_to_device(np.zeros(BINS, dtype=np.uint32))
                cuda.synchronize()
                t0 = time.perf_counter()
                launch(kernel, d_data, d_hist, blocks)
                cuda.synchronize()
                seconds += time.perf_counter() - t0
            seconds /= reps
            print(f"  {dist:<8} {name:<7} {seconds * 1e3:9.3f} ms  {n / seconds / 1e9:8.1f} GB/s")
    print("OK")
