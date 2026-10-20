import numpy as np
from numba import cuda


@cuda.jit
def whoami(out, block, thread):
    # cuda.grid(1) is blockIdx.x * blockDim.x + threadIdx.x.
    i = cuda.grid(1)
    # The last block can have threads past the end of the array.
    if i < out.size:
        out[i] = i
        block[i] = cuda.blockIdx.x
        thread[i] = cuda.threadIdx.x


@cuda.jit
def whoami2d(out):
    # x is the fast-moving dimension, so it maps to columns.
    col, row = cuda.grid(2)
    height, width = out.shape
    if col < width and row < height:
        out[row, col] = row * width + col


def run(n, threads):
    blocks = (n + threads - 1) // threads
    out = cuda.device_array(n, dtype=np.int32)
    block = cuda.device_array(n, dtype=np.int32)
    thread = cuda.device_array(n, dtype=np.int32)
    whoami[blocks, threads](out, block, thread)
    return out.copy_to_host(), block.copy_to_host(), thread.copy_to_host()


def run2d(width, height, block=(4, 2)):
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    out = cuda.to_device(np.full((height, width), -1, dtype=np.int32))
    whoami2d[grid, block](out)
    return out.copy_to_host()


if __name__ == "__main__":
    out, block, thread = run(20, 8)
    print(f"{'i':>5} {'block':>6} {'thread':>7}")
    for i, b, t in zip(out, block, thread):
        print(f"{i:>5} {b:>6} {t:>7}")
