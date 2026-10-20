import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def load(path, name):
    # numba/occupancy.py and triton/occupancy.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calc = load("occupancy_calc.py", "ch05_occupancy_calc")
nb = load("numba/occupancy.py", "ch05_numba_occupancy")
tr = load("triton/occupancy.py", "ch05_triton_occupancy")

T4, A100, H100 = (7, 5), (8, 0), (9, 0)
KB = 1024

# The same hand-computed cases as site/src/sims/occupancy/model.test.ts:
# (cc, regs per thread, shared bytes per block, threads per block, blocks per SM, occupancy).
CASES = [
    (A100, 32, 0, 256, 8, 1.0),  # warps 64/8 = 8; registers 16 warps per partition x 4 / 8 = 8
    (A100, 33, 0, 256, 6, 0.75),  # 1056 -> 1280 per warp; 16384 // 1280 = 12 per partition
    (A100, 32, 0, 32, 32, 0.5),  # 32 block slots of 1 warp each
    (T4, 32, 0, 64, 16, 1.0),  # 16 block slots x 2 warps = 32 warps
    (A100, 32, 48 * KB, 128, 3, 12 / 64),  # 49152 + 1024 = 50176; 167936 // 50176 = 3
    (H100, 64, 100 * KB, 256, 2, 0.25),  # 102400 + 1024 = 103424; 233472 // 103424 = 2
    (A100, 64, 0, 1024, 1, 0.5),  # 2048 x 32 = 65536, exactly the per-block limit
    (T4, 64, 0, 1024, 1, 1.0),
    (A100, 65, 0, 1024, 0, 0.0),  # 2304 x 32 = 73728 > 65536: launch fails
    (A100, 80, 0, 800, 0, 0.0),  # 25 warps rounded up to 28 x 2560 = 71680 > 65536
    (A100, 64, 0, 992, 1, 31 / 64),  # 31 warps rounded up to 32 x 2048 = 65536 fits
    (A100, 32, 163 * KB, 256, 1, 8 / 64),
    (A100, 32, 164 * KB, 256, 0, 0.0),  # 164 KB + 1 KB reserved > 163 KB + 1 KB
    (T4, 32, 64 * KB, 256, 1, 8 / 32),  # nothing reserved on Turing
    # The Best Practices Guide's example (cc 7.0, same register rules as 8.0): 37 regs -> 1280 per warp.
    (A100, 37, 0, 128, 12, 0.75),
    (A100, 37, 0, 320, 4, 40 / 64),
]


@pytest.mark.parametrize("cc, regs, smem, block, blocks, occ", CASES)
def test_calculator(cc, regs, smem, block, blocks, occ):
    assert calc.blocks_per_sm(cc, regs, smem, block)[0] == blocks
    assert calc.theoretical_occupancy(cc, regs, smem, block) == occ


def test_shared_memory_granularity():
    # T4 rounds 1000 bytes up to 1024; A100 rounds 1000 + 1024 reserved up to 2048.
    assert calc.blocks_per_sm(T4, 32, 1000, 256)[1]["shared"] == 64
    assert calc.blocks_per_sm(A100, 32, 1000, 256)[1]["shared"] == 82


def inputs(n):
    rng = np.random.default_rng(n)
    return rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)


@pytest.mark.parametrize("n", [1, 1000, 4097])
@pytest.mark.parametrize("kernel", ["saxpy", "saxpy_32", "saxpy_bounded"])
def test_numba(n, kernel):
    x, y = inputs(n)
    np.testing.assert_allclose(nb.run(2.0, x, y, getattr(nb, kernel)), np.float32(2) * x + y, rtol=1e-6)


def test_numba_grid_stride():
    # Fewer threads than elements: each thread must loop.
    x, y = inputs(3000)
    np.testing.assert_allclose(nb.run(2.0, x, y, threads=64, blocks=4), np.float32(2) * x + y, rtol=1e-6)


@pytest.mark.parametrize("n", [1, 1000, 4097])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_triton(n, num_warps):
    x, y = (torch.from_numpy(a).to(tr.DEVICE) for a in inputs(n))
    expected = 2.0 * x + y
    tr.saxpy(2.0, x, y, num_warps=num_warps)
    torch.testing.assert_close(y, expected)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary_matches_calculator():
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "occupancy"), "100000"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout

    major, minor = map(int, re.search(r"cc=(\d+)\.(\d+)", result.stdout).groups())
    if (major, minor) not in calc.LIMITS:
        pytest.skip(f"compute capability {major}.{minor} is not in the calculator's table")
    kernels = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"kernel (\w+) regs=(\d+) static_smem=(\d+)", result.stdout)}
    rows = re.findall(r"occupancy kernel=(\w+) smem=(\d+) block=(\d+) blocks=(\d+)", result.stdout)
    assert rows
    for name, smem, block, blocks in rows:
        regs, static = kernels[name]
        expected = calc.blocks_per_sm((major, minor), regs, static + int(smem), int(block))[0]
        assert int(blocks) == expected, f"{name} smem={smem} block={block}"
