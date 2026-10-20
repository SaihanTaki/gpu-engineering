import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/bitonic.py", "ch17_numba_bitonic")
tr = load("triton/block_sort.py", "ch17_triton_block_sort")

# Powers of two and sizes that need padding, with duplicates and negatives.
SIZES = [1, 2, 7, 64, 100, 256, 1000, 1024]


def keys(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-50, 50, n).astype(np.float32)


# The simulator runs one Python thread per GPU thread, so it gets a small tile and fewer sizes.
# Tile 32 with n = 300 (padded to 512) still runs global j >= tile stages between shared passes.
TILES = [32] if SIM else [32, nb.TILE]
NUMBA_SIZES = [1, 2, 7, 64, 100, 300] if SIM else SIZES + [5000]


@pytest.mark.parametrize("method", list(nb.METHODS))
@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba(method, tile, n):
    x = keys(n)
    np.testing.assert_array_equal(nb.sort(x, method, tile), np.sort(x))


@pytest.mark.parametrize("pattern", ["sorted", "reversed", "constant"])
def test_numba_patterns(pattern):
    x = {"sorted": np.arange(100), "reversed": np.arange(100)[::-1], "constant": np.zeros(100)}[pattern]
    np.testing.assert_array_equal(nb.sort(x, "shared", 32), np.sort(x).astype(np.float32))


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("n", SIZES + [2049])
def test_triton_block_sort(block, n):
    x = keys(n, seed=1)
    out = tr.block_sort(torch.from_numpy(x).to(tr.DEVICE), block).cpu().numpy()
    ref = np.concatenate([np.sort(x[i : i + block]) for i in range(0, n, block)])
    np.testing.assert_array_equal(out, ref)


NUMBA_PTX = """
from numba import cuda, float32, int32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
step, _ = cuda.compile_ptx(m.bitonic_step.py_func, (float32[::1], int32, int32), cc=(8, 0))
shared, _ = cuda.compile_ptx(m.bitonic_shared.py_func, (float32[::1], int32, int32), cc=(8, 0))
print(step.count("bar.sync"), shared.count("bar.sync"), shared.count("st.shared"))
"""


def test_numba_compiles_for_gpu():
    # The simulator compiles nothing, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/bitonic.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    step_bars, shared_bars, shared_stores = map(int, res.stdout.split())
    assert step_bars == 0
    assert shared_bars > 0 and shared_stores > 0


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/bitonic_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"bitonic_kernels.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        pytest.skip(f"NVRTC unavailable: {err}")
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, log.decode()
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(prog, ptx)
    ptx = ptx.decode()
    assert ".entry _Z12bitonic_stepPfiii" in ptx
    # 1024 floats of shared memory per block.
    assert ".shared .align 4 .b8 _ZZ14bitonic_sharedPfiiE1s[4096]" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 1000, 1024, 4097, 1 << 20])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "bitonic"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
