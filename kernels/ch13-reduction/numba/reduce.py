import os
import sys
import time

import numpy as np
from numba import cuda, float32

# A power of two: every tree level halves the number of values.
BLOCK = 256
FULL_MASK = 0xFFFFFFFF


@cuda.jit
def reduce_interleaved(x, out):
    sdata = cuda.shared.array(BLOCK, float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    sdata[tid] = x[i] if i < x.size else float32(0)
    cuda.syncthreads()
    s = 1
    while s < BLOCK:
        # The working threads are spread over every warp, so every warp diverges.
        if tid % (2 * s) == 0:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s *= 2
    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]


@cuda.jit
def reduce_sequential(x, out):
    sdata = cuda.shared.array(BLOCK, float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    sdata[tid] = x[i] if i < x.size else float32(0)
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 0:
        # Threads 0 to s - 1 work: whole warps until s drops below 32.
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        out[cuda.blockIdx.x] = sdata[0]


@cuda.jit
def reduce_warp_final(x, out):
    sdata = cuda.shared.array(BLOCK, float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    sdata[tid] = x[i] if i < x.size else float32(0)
    cuda.syncthreads()
    s = BLOCK // 2
    while s > 32:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid < 32:
        # The last 64 values fit in one warp: finish in registers, with no barrier.
        v = sdata[tid] + sdata[tid + 32]
        offset = 16
        while offset > 0:
            v += cuda.shfl_down_sync(FULL_MASK, v, offset)
            offset //= 2
        if tid == 0:
            out[cuda.blockIdx.x] = v


@cuda.jit(device=True)
def warp_sum(v):
    # After offsets 16, 8, 4, 2 and 1, lane 0 holds the sum of all 32 lanes.
    offset = 16
    while offset > 0:
        v += cuda.shfl_down_sync(FULL_MASK, v, offset)
        offset //= 2
    return v


@cuda.jit
def reduce_atomic(x, out):
    warp_sums = cuda.shared.array(32, float32)
    tid = cuda.threadIdx.x
    lane = tid % 32
    warp = tid // 32
    v = float32(0)
    # Grid-stride loop: each thread sums many elements in a register first.
    for i in range(cuda.grid(1), x.size, cuda.gridsize(1)):
        v += x[i]
    v = warp_sum(v)
    if lane == 0:
        warp_sums[warp] = v
    cuda.syncthreads()
    if warp == 0:
        v = warp_sums[lane] if lane < BLOCK // 32 else float32(0)
        v = warp_sum(v)
        # One atomic per block, not one per element.
        if lane == 0:
            cuda.atomic.add(out, 0, v)


MULTI_PASS = {"interleaved": reduce_interleaved, "sequential": reduce_sequential, "warp_final": reduce_warp_final}
# The Numba CUDA simulator has no shfl_down_sync, so these two need a GPU.
NEEDS_SHUFFLE = {"warp_final", "atomic"}


def partial_buffers(n):
    blocks = (n + BLOCK - 1) // BLOCK
    return cuda.device_array(blocks, dtype=np.float32), cuda.device_array(blocks, dtype=np.float32)


def multi_pass(kernel, d_x, bufs=None):
    # Each launch leaves one partial sum per block; launch again on those,
    # ping-ponging between two buffers so nothing is allocated per pass.
    d_a, d_b = bufs or partial_buffers(d_x.size)
    while True:
        blocks = (d_x.size + BLOCK - 1) // BLOCK
        d_out = d_a[:blocks]
        kernel[blocks, BLOCK](d_x, d_out)
        if blocks == 1:
            return d_out
        d_x = d_out
        d_a, d_b = d_b, d_a


def one_pass_atomic(d_x, blocks=None):
    if blocks is None:
        blocks = 4 * cuda.get_current_device().MULTIPROCESSOR_COUNT
    d_out = cuda.to_device(np.zeros(1, dtype=np.float32))
    reduce_atomic[blocks, BLOCK](d_x, d_out)
    return d_out


def total(name, x):
    d_x = cuda.to_device(x)
    d_out = one_pass_atomic(d_x) if name == "atomic" else multi_pass(MULTI_PASS[name], d_x)
    return float(d_out.copy_to_host()[0])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 22
    # Whole numbers from -3 to 3: the float32 sum is exact, so every order agrees.
    x = np.random.default_rng(0).integers(-3, 4, n).astype(np.float32)
    expected = int(x.astype(np.int64).sum())
    d_x = cuda.to_device(x)
    bufs = partial_buffers(n)
    reps = 20
    print(f"n = {n} floats, {BLOCK} threads per block, expected sum {expected}")
    for name in [*MULTI_PASS, "atomic"]:
        if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" and name in NEEDS_SHUFFLE:
            print(f"  {name:<11} skipped: needs a GPU")
            continue
        run = (lambda: one_pass_atomic(d_x)) if name == "atomic" else (lambda: multi_pass(MULTI_PASS[name], d_x, bufs))
        # The first call compiles the kernel, so it is not timed.
        run()
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            d_out = run()
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        got = float(d_out.copy_to_host()[0])
        assert got == expected, (name, got, expected)
        print(f"  {name:<11} {seconds * 1e3:9.3f} ms  {4 * n / seconds / 1e9:8.1f} GB/s")
    print("OK")
