import sys
import time

import numpy as np
from numba import config, cuda

WARP = 32
BLOCK = 128


@cuda.jit
def spmv_csr_scalar(row_ptr, col, val, x, y):
    # One thread per row.
    row = cuda.grid(1)
    if row < y.size:
        s = np.float32(0.0)
        # Each thread loops as long as its own row; the warp waits for the longest.
        for i in range(row_ptr[row], row_ptr[row + 1]):
            s += val[i] * x[col[i]]
        y[row] = s


@cuda.jit
def spmv_csr_vector(row_ptr, col, val, x, y):
    # One warp per row. BLOCK is a multiple of 32.
    lane = cuda.threadIdx.x % WARP
    row = cuda.grid(1) // WARP
    # row is the same for all 32 lanes, so whole warps leave and every shuffle sees all lanes.
    if row >= y.size:
        return
    s = np.float32(0.0)
    # Neighbouring lanes read neighbouring entries of col and val.
    for i in range(row_ptr[row] + lane, row_ptr[row + 1], WARP):
        s += val[i] * x[col[i]]
    offset = WARP // 2
    while offset > 0:
        s += cuda.shfl_down_sync(0xFFFFFFFF, s, offset)
        offset //= 2
    if lane == 0:
        y[row] = s


@cuda.jit
def spmv_ell(col, val, x, y):
    # One thread per row. col and val have shape (width, rows): column-major ELL.
    row = cuda.grid(1)
    if row < y.size:
        s = np.float32(0.0)
        # At each k, neighbouring threads read neighbouring addresses.
        for k in range(col.shape[0]):
            s += val[k, row] * x[col[k, row]]
        y[row] = s


def csr_from_dense(a):
    rows, cols = np.nonzero(a)
    row_ptr = np.zeros(a.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=a.shape[0]), out=row_ptr[1:])
    return row_ptr, cols.astype(np.int32), a[rows, cols].astype(np.float32)


def ell_from_csr(row_ptr, col, val):
    lengths = np.diff(row_ptr)
    width = int(lengths.max(initial=0))
    rows = lengths.size
    # Padded slots get column 0 and value 0, so they add nothing.
    ell_col = np.zeros((width, rows), dtype=np.int32)
    ell_val = np.zeros((width, rows), dtype=np.float32)
    row_of = np.repeat(np.arange(rows), lengths)
    slot = np.arange(col.size) - row_ptr[row_of]
    ell_col[slot, row_of] = col
    ell_val[slot, row_of] = val
    return ell_col, ell_val


KERNELS = ("csr_scalar", "csr_vector", "ell")


def launch(name, a, x, y):
    # `a` is (row_ptr, col, val) on the device for CSR, (ell_col, ell_val) for ELL.
    rows = y.size
    if name == "csr_scalar":
        spmv_csr_scalar[(rows + BLOCK - 1) // BLOCK, BLOCK](*a, x, y)
    elif name == "csr_vector":
        spmv_csr_vector[(rows * WARP + BLOCK - 1) // BLOCK, BLOCK](*a, x, y)
    else:
        spmv_ell[(rows + BLOCK - 1) // BLOCK, BLOCK](*a, x, y)


def device_matrix(name, row_ptr, col, val):
    host = ell_from_csr(row_ptr, col, val) if name == "ell" else (row_ptr, col, val)
    return tuple(cuda.to_device(np.ascontiguousarray(h)) for h in host)


def run(name, row_ptr, col, val, x):
    a = device_matrix(name, row_ptr, col, val)
    y = cuda.device_array(row_ptr.size - 1, dtype=np.float32)
    launch(name, a, cuda.to_device(x.astype(np.float32)), y)
    return y.copy_to_host()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    reps = 20
    rng = np.random.default_rng(0)
    # Rows of 1 to 16 nonzeros in random columns.
    lengths = rng.integers(1, 17, rows)
    row_ptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    col = rng.integers(0, rows, row_ptr[-1]).astype(np.int32)
    val = rng.random(row_ptr[-1], dtype=np.float32)
    x = rng.random(rows, dtype=np.float32)
    row_of = np.repeat(np.arange(rows), lengths)
    want = np.zeros(rows)
    np.add.at(want, row_of, val.astype(np.float64) * x[col])

    print(f"{rows} rows, {row_ptr[-1]} nonzeros")
    d_x = cuda.to_device(x)
    d_y = cuda.device_array(rows, dtype=np.float32)
    for name in KERNELS:
        if name == "csr_vector" and config.ENABLE_CUDASIM:
            print(f"  {name:<11} skipped: the simulator has no warp shuffles")
            continue
        a = device_matrix(name, row_ptr, col, val)
        # The first call compiles the kernel, so it is not timed.
        launch(name, a, d_x, d_y)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            launch(name, a, d_x, d_y)
        cuda.synchronize()
        seconds = (time.perf_counter() - t0) / reps
        np.testing.assert_allclose(d_y.copy_to_host(), want, rtol=1e-4, atol=1e-4)
        print(f"  {name:<11} {seconds * 1e3:9.3f} ms")
    print("OK")
