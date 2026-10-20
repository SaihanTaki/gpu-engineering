import os
import sys
from pathlib import Path

import numpy as np
from numba import cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy_calc import LIMITS, blocks_per_sm  # noqa: E402

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def saxpy_body(a, x, y):
    # The grid-stride loop lets any grid size cover all elements.
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for i in range(start, y.size, stride):
        y[i] = a * x[i] + y[i]


# An explicit signature compiles at once, so the resource queries return one number.
SIG = "void(float32, float32[:], float32[:])"
saxpy = cuda.jit(SIG)(saxpy_body)
# The simulator's jit takes no register options, so only real GPUs get the limits.
if SIM:
    saxpy_32 = saxpy_bounded = saxpy
else:
    saxpy_32 = cuda.jit(SIG, max_registers=32)(saxpy_body)
    # At most 256 threads per block, at least 8 such blocks resident per SM.
    saxpy_bounded = cuda.jit(SIG, launch_bounds=(256, 8))(saxpy_body)


def run(a, x, y, kernel=saxpy, threads=256, blocks=None):
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    if blocks is None:
        blocks = (y.size + threads - 1) // threads
    kernel[blocks, threads](np.float32(a), d_x, d_y)
    return d_y.copy_to_host()


def report(name, kernel):
    regs = kernel.get_regs_per_thread()
    smem = kernel.get_shared_mem_per_block()
    cc = cuda.get_current_device().compute_capability
    print(f"{name}: {regs} registers per thread, {smem} bytes static shared memory")
    if cc not in LIMITS:
        print(f"  compute capability {cc} is not in occupancy_calc.LIMITS")
        return
    for block in (64, 128, 256, 512, 1024):
        blocks, limits = blocks_per_sm(cc, regs, smem, block)
        warps = blocks * block // 32
        limiting = [r for r, v in limits.items() if v == blocks]
        print(f"  block {block:4}: {blocks:2} blocks, {warps:2} of {LIMITS[cc]['max_warps']} warps,"
              f" limited by {' and '.join(limiting)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    rng = np.random.default_rng(0)
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)
    expected = np.float32(2.0) * x + y

    for name, kernel in [("saxpy", saxpy), ("saxpy_32", saxpy_32), ("saxpy_bounded", saxpy_bounded)]:
        np.testing.assert_allclose(run(2.0, x, y, kernel), expected, rtol=1e-6)
        if not SIM:
            report(name, kernel)
    if SIM:
        print("Simulator: no register counts. Run on a GPU to see them.")
    print("OK")
