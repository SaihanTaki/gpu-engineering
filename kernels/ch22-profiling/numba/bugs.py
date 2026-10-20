import sys

import numpy as np
from numba import cuda, uint32

THREADS = 256


@cuda.jit
def count_racy(x, total):
    count = cuda.shared.array(1, uint32)
    if cuda.threadIdx.x == 0:
        count[0] = 0
    cuda.syncthreads()
    i = cuda.grid(1)
    if i < x.size and x[i] > 0:
        # A load, an add and a store: another thread can load between them.
        count[0] += 1
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(total, 0, count[0])


@cuda.jit
def count_atomic(x, total):
    count = cuda.shared.array(1, uint32)
    if cuda.threadIdx.x == 0:
        count[0] = 0
    cuda.syncthreads()
    i = cuda.grid(1)
    if i < x.size and x[i] > 0:
        cuda.atomic.add(count, 0, 1)
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(total, 0, count[0])


@cuda.jit
def scale_unguarded(x, out):
    i = cuda.grid(1)
    # No `if i < x.size`: the last block's extra threads index past the end.
    out[i] = 2 * x[i]


@cuda.jit
def scale(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = 2 * x[i]


@cuda.jit
def diff_wrap(x, out):
    i = cuda.grid(1)
    if i < x.size:
        # At i = 0 this reads x[-1]. The simulator follows NumPy and wraps to the last element.
        out[i] = x[i] - x[i - 1]


COUNTERS = {"racy": count_racy, "atomic": count_atomic}


def count(name, x, threads=THREADS):
    d_total = cuda.to_device(np.zeros(1, dtype=np.uint32))
    blocks = (x.size + threads - 1) // threads
    COUNTERS[name][blocks, threads](cuda.to_device(x), d_total)
    return int(d_total.copy_to_host()[0])


def run(kernel, x, threads=THREADS):
    out = cuda.to_device(np.zeros_like(x))
    blocks = (x.size + threads - 1) // threads
    kernel[blocks, threads](cuda.to_device(x), out)
    return out.copy_to_host()


def make_input(n, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    x = make_input(n)
    print(f"{n} elements, {int((x > 0).sum())} positive")
    for name in COUNTERS:
        print(f"  count_{name:<7} counted {count(name, x)}")
    print(f"  scale            {'OK' if np.array_equal(run(scale, x), 2 * x) else 'WRONG'}")
    try:
        # On a GPU this writes past the end without a word; the simulator raises IndexError.
        run(scale_unguarded, x)
        print("  scale_unguarded  no error reported")
    except IndexError as e:
        print(f"  scale_unguarded  IndexError: {e}")
