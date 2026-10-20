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
    # numba/reduce.py and triton/reduce.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/reduce.py", "ch13_numba_reduce")
tr = load("triton/reduce.py", "ch13_triton_reduce")

# One element, one partial block, one exact block, a ragged second pass, and three passes.
SIZES = [1, 100, 256, 1000, 70_000]
# The Numba simulator runs one Python thread per GPU thread, so it gets the smaller sizes.
NUMBA_SIZES = SIZES[:-1] if SIM else SIZES


def whole_numbers(n):
    # Integers from -3 to 3: every partial sum is exact in float32, in any order.
    return np.random.default_rng(n).integers(-3, 4, n).astype(np.float32)


def exact(x):
    return float(x.astype(np.int64).sum())


@pytest.mark.parametrize("n", NUMBA_SIZES)
@pytest.mark.parametrize("name", [*nb.MULTI_PASS, "atomic"])
def test_numba(name, n):
    if SIM and name in nb.NEEDS_SHUFFLE:
        pytest.skip("the Numba CUDA simulator has no shfl_down_sync")
    x = whole_numbers(n)
    assert nb.total(name, x) == exact(x)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("method", list(tr.METHODS))
def test_triton(method, n):
    x = whole_numbers(n)
    got = tr.METHODS[method](torch.from_numpy(x).to(tr.DEVICE))
    assert float(got) == exact(x)


@pytest.mark.parametrize("method", list(tr.METHODS))
def test_triton_random_floats(method):
    # Real-valued data: the tree adds in a different order than NumPy, so compare with a tolerance.
    x = np.random.default_rng(1).standard_normal(50_000).astype(np.float32)
    got = float(tr.METHODS[method](torch.from_numpy(x).to(tr.DEVICE)))
    np.testing.assert_allclose(got, x.astype(np.float64).sum(), rtol=1e-4, atol=1e-3)


def test_triton_atomic_with_many_programs():
    # More programs than the grid-stride loop needs: some programs add 0.
    x = whole_numbers(5000)
    assert float(tr.sum_atomic(torch.from_numpy(x).to(tr.DEVICE), programs=64)) == exact(x)


NUMBA_PTX = """
from numba import cuda, float32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
def show(name, ptx):
    print(name, ptx.count("shfl.sync.down"), ptx.count("bar.sync"), ptx.count("atom.global.add.f32") + ptx.count("red.global.add.f32"))
for name, k in m.MULTI_PASS.items():
    show(name, cuda.compile_ptx(k.py_func, (float32[::1], float32[::1]), cc=(8, 0))[0])
show("warp_sum", cuda.compile_ptx(m.warp_sum.py_func, (float32,), device=True, cc=(8, 0))[0])
try:
    show("atomic", cuda.compile_ptx(m.reduce_atomic.py_func, (float32[::1], float32[::1]), cc=(8, 0))[0])
except cuda.CudaSupportError:
    pass  # Numba compiles a called device function for the current GPU, so this kernel needs one.
"""


def test_numba_compiles_for_gpu():
    # The simulator compiles nothing and cannot run the shuffle kernels, so compile them to PTX in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/reduce.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    for name in ("interleaved", "sequential"):
        assert counts[name][0] == 0 and counts[name][1] > 0, counts
    # Five shuffles, one per offset, once the compiler unrolls the loop; at least one if it does not.
    assert counts["warp_final"][0] > 0 and counts["warp_sum"][0] > 0, counts
    if "atomic" in counts:
        assert counts["atomic"][0] > 0 and counts["atomic"][2] > 0, counts


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/reduce_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"reduce_kernels.cu", 0, [], [])
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
    for name in ("reduce_interleaved", "reduce_sequential", "reduce_warp_final", "reduce_atomic"):
        assert f".entry _Z{len(name)}{name}PKfPfj" in ptx
    assert "shfl.sync.down" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", SIZES)
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "reduce"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
