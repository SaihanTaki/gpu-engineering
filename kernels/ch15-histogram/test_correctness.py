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


def load(path, name):
    # numba/histogram.py and triton/histogram.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/histogram.py", "ch15_numba_histogram")
tr = load("triton/histogram.py", "ch15_triton_histogram")

DISTS = ["uniform", "skewed", "one-bin"]
# One element, a partial block, and sizes past one block and one grid-stride pass.
SIZES = [1, 1000, 5000]
# The Numba simulator runs one Python thread per GPU thread, so it gets a small grid.
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
NUMBA_LAUNCH = (2, 64) if SIM else (16, 256)


def reference(data):
    return np.bincount(data, minlength=nb.BINS)


@pytest.mark.parametrize("name", list(nb.KERNELS))
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("n", SIZES)
def test_numba(name, dist, n):
    data = nb.make_input(dist, n)
    blocks, threads = NUMBA_LAUNCH
    out = nb.run(name, data, blocks, threads)
    np.testing.assert_array_equal(out, reference(data))
    assert out.sum() == n


@pytest.mark.parametrize("variant", tr.KERNELS)
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("n", SIZES + [3 * tr.BLOCK + 1])
def test_triton(variant, dist, n):
    data = nb.make_input(dist, n)
    out = tr.histogram(torch.from_numpy(data).to(tr.DEVICE), variant)
    np.testing.assert_array_equal(out.cpu().numpy(), reference(data))


NUMBA_PTX = """
from numba import cuda, uint8, uint32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sig = (uint8[::1], uint32[::1])
for name, k in m.KERNELS.items():
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 0))
    print(name, ptx.count("atom.global"), ptx.count("atom.shared"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/histogram.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    assert counts["global"][0] > 0 and counts["global"][1] == 0
    assert counts["shared"][0] > 0 and counts["shared"][1] > 0


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/histogram_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"histogram_kernels.cu", 0, [], [])
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
    # The private copy is 256 unsigned ints in shared memory.
    assert ".shared .align 4 .b8 _ZZ11hist_sharedPKhiPjE5local[1024]" in ptx
    # One global atomic in hist_global; a shared one and a global one in hist_shared.
    assert ptx.count("atom.global.add.u32") == 2
    assert ptx.count("atom.shared.add.u32") == 1


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 1000, 1 << 20])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "histogram"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
