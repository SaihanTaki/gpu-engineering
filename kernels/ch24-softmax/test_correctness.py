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
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
EPS = 1e-5


def load(path, name):
    # numba/softmax.py and triton/softmax.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/softmax.py", "ch24_numba_softmax")
tr = load("triton/softmax.py", "ch24_triton_softmax")

# One column, a ragged row shorter than a block, exactly one block of 128, several blocks with a ragged end.
SHAPES = [(3, 1), (4, 100), (2, 128), (3, 1000)]
# The Numba simulator runs one Python thread per GPU thread, so it gets smaller inputs.
NUMBA_SHAPES = SHAPES[:3] if SIM else SHAPES


def rows(shape, seed=0):
    x = 3 * np.random.default_rng(seed).standard_normal(shape)
    return x.astype(np.float32)


def stress(cols):
    # Row 0: one value whose exp overflows float32 (exp(1000) is about 1e434; float32 stops near 3.4e38).
    # Row 1: every value huge and equal. Row 2: very negative values, whose exp underflows to 0.
    x = rows((3, cols), seed=cols)
    x[0, cols // 2] = 1000.0
    x[1] = 1e4
    x[2] -= 200.0
    return x


def softmax_ref(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def layernorm_ref(x, w, b):
    x = x.astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    rstd = 1 / np.sqrt(var + EPS)
    return (x - mean) * rstd * w + b, mean[:, 0], rstd[:, 0]


def affine(cols):
    rng = np.random.default_rng(cols + 1)
    return (1 + 0.1 * rng.standard_normal(cols)).astype(np.float32), (0.1 * rng.standard_normal(cols)).astype(np.float32)


def offset(shape):
    # LayerNorm input with a large mean and a small spread, where E[x^2] - E[x]^2 loses its digits in float32.
    return (1000 + rows(shape, seed=5) / 3).astype(np.float32)


SM_TOL = dict(rtol=1e-5, atol=1e-7)
LN_TOL = dict(rtol=1e-4, atol=1e-3)


def test_naive_softmax_overflows():
    # The failure the safe and online kernels avoid: exp(1000) is inf in float32, and inf / inf is NaN.
    x = stress(8)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)
        y = e / e.sum()
    assert np.isinf(e).any() and np.isnan(y).any()


def test_naive_variance_cancels():
    # The failure Welford's update avoids: E[x^2] - E[x]^2 in float32 for values near 1000 with spread 1.
    x = offset((1, 4096))[0]
    naive = np.mean(x * x, dtype=np.float32) - np.float32(np.mean(x, dtype=np.float32)) ** 2
    exact = x.astype(np.float64).var()
    assert abs(naive - exact) / exact > 0.01


@pytest.mark.parametrize("shape", NUMBA_SHAPES)
@pytest.mark.parametrize("method", list(nb.SOFTMAX))
def test_numba_softmax(method, shape):
    x = rows(shape)
    np.testing.assert_allclose(nb.softmax(x, method), softmax_ref(x), **SM_TOL)


@pytest.mark.parametrize("method", list(nb.SOFTMAX))
def test_numba_softmax_stable(method):
    x = stress(200)
    y = nb.softmax(x, method)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, softmax_ref(x), **SM_TOL)


@pytest.mark.parametrize("shape", NUMBA_SHAPES)
def test_numba_layernorm(shape):
    x = rows(shape)
    w, b = affine(shape[1])
    y, mean, rstd = nb.layernorm(x, w, b, EPS)
    want, want_mean, want_rstd = layernorm_ref(x, w, b)
    np.testing.assert_allclose(y, want, **LN_TOL)
    np.testing.assert_allclose(mean, want_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rstd, want_rstd, rtol=1e-4)


def test_numba_layernorm_large_mean():
    x = offset((2, 300))
    w, b = affine(300)
    np.testing.assert_allclose(nb.layernorm(x, w, b, EPS)[0], layernorm_ref(x, w, b)[0], **LN_TOL)


@pytest.mark.parametrize("shape", SHAPES)
def test_triton_softmax(shape):
    x = rows(shape)
    y = tr.softmax(torch.from_numpy(x).to(tr.DEVICE)).cpu().numpy()
    np.testing.assert_allclose(y, softmax_ref(x), **SM_TOL)


def test_triton_softmax_stable():
    x = stress(1000)
    y = tr.softmax(torch.from_numpy(x).to(tr.DEVICE)).cpu().numpy()
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, softmax_ref(x), **SM_TOL)


@pytest.mark.parametrize("shape", SHAPES)
def test_triton_layernorm(shape):
    x = rows(shape)
    w, b = affine(shape[1])
    dev = lambda a: torch.from_numpy(a).to(tr.DEVICE)
    y, mean, rstd = tr.layernorm(dev(x), dev(w), dev(b), EPS)
    want, want_mean, want_rstd = layernorm_ref(x, w, b)
    np.testing.assert_allclose(y.cpu().numpy(), want, **LN_TOL)
    np.testing.assert_allclose(mean.cpu().numpy(), want_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rstd.cpu().numpy(), want_rstd, rtol=1e-4)
    # PyTorch's own layer_norm, as a second reference.
    ref = torch.nn.functional.layer_norm(dev(x), (shape[1],), dev(w), dev(b), EPS)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


def test_triton_layernorm_large_mean():
    x = offset((2, 1000))
    w, b = affine(1000)
    dev = lambda a: torch.from_numpy(a).to(tr.DEVICE)
    y = tr.layernorm(dev(x), dev(w), dev(b), EPS)[0].cpu().numpy()
    np.testing.assert_allclose(y, layernorm_ref(x, w, b)[0], **LN_TOL)


NUMBA_PTX = """
from numba import cuda, float32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
mat = float32[:, ::1]
vec = float32[::1]
for name, k in m.SOFTMAX.items():
    cuda.compile_ptx(k.py_func, (mat, mat), cc=(8, 0))
    print(name)
cuda.compile_ptx(m.layernorm_welford.py_func, (mat, vec, vec, mat, vec, vec, float32), cc=(8, 0))
print("layernorm")
"""


def test_numba_compiles_for_gpu():
    # The simulator compiles nothing, so compile the kernels to PTX in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/softmax.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and ("NVVM" in res.stderr or "CudaSupportError" in res.stderr):
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["safe", "online", "layernorm"]


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/softmax_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"softmax_kernels.cu", 0, [], [])
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
    entries = re.findall(r"\.entry (\w+)", ptx.decode())
    assert entries == ["softmax_safe", "softmax_online", "softmax_online_warp", "layernorm_welford"]
    assert "shfl.sync.bfly" in ptx.decode()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("shape", [(2, 1), (64, 100), (300, 1024), (8, 5000)])
def test_cuda_binary(shape):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "softmax"), *map(str, shape)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
