import numpy as np
from numba import cuda


@cuda.jit
def saxpy(a, x, y, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a * x[i] + y[i]


def run(a, x, y, threads=256):
    blocks = (x.size + threads - 1) // threads
    out = cuda.device_array_like(x)
    saxpy[blocks, threads](np.float32(a), cuda.to_device(x), cuda.to_device(y), out)
    return out.copy_to_host()


if __name__ == "__main__":
    x = np.arange(8, dtype=np.float32)
    y = np.ones(8, dtype=np.float32)
    print(run(2.0, x, y))
