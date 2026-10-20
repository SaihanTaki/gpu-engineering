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
    # numba/transpose.py and triton/transpose.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/transpose.py", "ch11_numba_transpose")
tr = load("triton/transpose.py", "ch11_triton_transpose")

# One element, one exact tile, ragged tiles in both directions, tall and wide, and several full tiles.
SHAPES = [(1, 1), (32, 32), (33, 65), (100, 37), (7, 130), (64, 96)]
# The Numba simulator runs one Python thread per GPU thread, so it gets the smaller shapes.
NUMBA_SHAPES = SHAPES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SHAPES


def matrix(shape):
    rows, cols = shape
    # Distinct values, so a swapped or missed element cannot go unnoticed.
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


@pytest.mark.parametrize("name", list(nb.KERNELS))
@pytest.mark.parametrize("shape", NUMBA_SHAPES)
def test_numba(name, shape):
    a = matrix(shape)
    np.testing.assert_array_equal(nb.run(name, a), a.T)


@pytest.mark.parametrize("variant", list(tr.KERNELS))
@pytest.mark.parametrize("shape", SHAPES)
def test_triton(variant, shape):
    a = matrix(shape)
    out = tr.transpose(torch.from_numpy(a).to(tr.DEVICE), variant)
    np.testing.assert_array_equal(out.cpu().numpy(), a.T)


NUMBA_PTX = """
from numba import cuda, float32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sig = (float32[:, ::1], float32[:, ::1])
for name, k in m.KERNELS.items():
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 0))
    print(name, ptx.count("ld.shared"), ptx.count("st.shared"), ptx.count("bar.sync"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/transpose.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    assert counts["naive"] == (0, 0, 0)
    for name in ("tiled", "padded"):
        assert all(n > 0 for n in counts[name]), counts


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/transpose_kernels.cuh").read_text()
    src += "\ntemplate __global__ void transpose_shared<0>(const float *, float *, int, int);\n"
    src += "template __global__ void transpose_shared<1>(const float *, float *, int, int);\n"
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"transpose_kernels.cu", 0, [], [])
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
    # The padded tile is 32 x 33 floats.
    assert ".b8 _ZZ16transpose_sharedILi0EEvPKfPfiiE4tile[4096]" in ptx
    assert ".b8 _ZZ16transpose_sharedILi1EEvPKfPfiiE4tile[4224]" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("shape", SHAPES)
def test_cuda_binary(shape):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(HERE / "cuda" / "transpose"), *map(str, shape)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
