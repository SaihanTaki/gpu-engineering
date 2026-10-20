import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
A = 1.5


def load(path, name):
    # numba/fusion.py and triton/fusion.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/fusion.py", "ch12_numba_fusion")
tr = load("triton/fusion.py", "ch12_triton_fusion")

# One element, a partial block, exact blocks of 256 and 1024, and several blocks with a ragged end.
SIZES = [1, 100, 256, 1024, 3000]


def inputs(n):
    rng = np.random.default_rng(n)
    return rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)


def reference(x, y):
    return np.maximum(np.float32(A) * x + y, np.float32(0))


# a * x + y may become one FMA in the fused kernels, which rounds once instead of twice.
TOL = dict(rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("version", ["unfused", "fused"])
@pytest.mark.parametrize("n", SIZES)
def test_numba(version, n):
    x, y = inputs(n)
    run = nb.unfused_host if version == "unfused" else nb.fused_host
    np.testing.assert_allclose(run(x, y, A), reference(x, y), **TOL)


@pytest.mark.parametrize("version", ["unfused", "fused"])
@pytest.mark.parametrize("n", SIZES)
def test_triton(version, n):
    x, y = inputs(n)
    run = tr.unfused if version == "unfused" else tr.fused
    out = run(torch.from_numpy(x).to(tr.DEVICE), torch.from_numpy(y).to(tr.DEVICE), A)
    np.testing.assert_allclose(out.cpu().numpy(), reference(x, y), **TOL)


def test_relu_clamps_negatives():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)
    np.testing.assert_array_equal(nb.fused_host(x, y, A), [0.0, 0.0, 3.0])


# Global loads and stores per kernel: the DRAM traffic the chapter counts, one float each.
TRAFFIC = {"scale": (1, 1), "add": (2, 1), "relu": (1, 1), "fused": (2, 1)}


def ptx_functions(ptx):
    # Split PTX into its .entry functions and keep the short kernel name.
    parts = re.split(r"\.visible \.entry ", ptx)[1:]
    out = {}
    for p in parts:
        name = p.split("(")[0]
        short = next(k for k in TRAFFIC if k in name)
        out[short] = (len(re.findall(r"\bld\.global", p)), len(re.findall(r"\bst\.global", p)))
    return out


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/fusion_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"fusion_kernels.cu", 0, [], [])
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
    assert ptx_functions(ptx.decode()) == TRAFFIC


NUMBA_PTX = """
import importlib.util, sys
from numba import cuda, float32
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
v = float32[::1]
sigs = {"scale": (v, v, float32), "add": (v, v, v), "relu": (v, v), "fused": (v, v, v, float32)}
for name, sig in sigs.items():
    ptx, _ = cuda.compile_ptx(getattr(m, name).py_func, sig, cc=(8, 0))
    print(name, ptx.count("ld.global"), ptx.count("st.global"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/fusion.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    assert counts == TRAFFIC


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 1000, 1 << 20])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "fusion"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
