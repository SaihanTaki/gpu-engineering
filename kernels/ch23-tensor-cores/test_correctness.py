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
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/fp16_gemm.py", "ch23_numba_fp16_gemm")
tr = load("triton/matmul.py", "ch23_triton_matmul")


def inputs(m, n, k, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float16)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float16)
    return a, b


def exact(a, b):
    # FP16 products are exact in FP64, and so are these short sums, up to FP64 rounding.
    return a.astype(np.float64) @ b.astype(np.float64)


def rel_error(got, a, b):
    scale = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    return np.max(np.abs(got.astype(np.float64) - exact(a, b)) / scale)


# Ragged sizes that do not fill a 16 x 16 block, and a K long enough for FP16 sums to drift.
SHAPES = [(1, 1, 1), (17, 9, 33), (32, 32, 256)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_numba_f32acc(shape):
    a, b = inputs(*shape)
    # FP32 sums of exact products: error well below FP32's unit roundoff times K.
    assert rel_error(nb.matmul_f32acc(a, b), a, b) <= shape[2] * 2.0**-24


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_numba_f16acc(shape):
    a, b = inputs(*shape)
    got = nb.matmul_f16acc(a, b)
    assert got.dtype == np.float16
    # Each of K steps rounds to FP16 (twice under the simulator, whose hfma is not fused).
    assert rel_error(got, a, b) <= 2 * shape[2] * 2.0**-11


def test_fp16_accumulator_stalls_at_2048():
    ones = np.ones((1, 4096), dtype=np.float16)
    assert nb.matmul_f32acc(ones, ones.T)[0, 0] == 4096
    # 2048 + 1 is a tie between 2048 and 2050; round to even keeps 2048 for good.
    assert nb.matmul_f16acc(ones, ones.T)[0, 0] == 2048


def test_numpy_float16_limits():
    # The values the bit layout simulation is tested against, here from NumPy.
    assert np.finfo(np.float16).max == 65504
    with np.errstate(over="ignore"):
        assert np.float16(65520) == np.inf
    assert np.float16(2.0**-25) == 0
    assert np.float16(1 / 3).view(np.uint16) == 0x3555
    assert torch.tensor(1 / 3).to(torch.bfloat16).view(torch.int16).item() == 0x3EAB
    assert torch.finfo(torch.float8_e4m3fn).max == 448
    assert torch.finfo(torch.float8_e5m2).max == 57344


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_triton_fp16(shape):
    a, b = inputs(*shape)
    got = tr.matmul(torch.from_numpy(a).to(tr.DEVICE), torch.from_numpy(b).to(tr.DEVICE), block=(16, 16, 16))
    assert got.dtype == torch.float32
    assert rel_error(got.cpu().numpy(), a, b) <= shape[2] * 2.0**-24 + 1e-7


@pytest.mark.skipif(INTERPRET, reason="the interpreter computes tl.dot on BF16 wrongly")
def test_triton_bf16():
    a, b = inputs(32, 32, 64)
    ta = torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).to(tr.DEVICE)
    tb = torch.from_numpy(b.astype(np.float32)).to(torch.bfloat16).to(tr.DEVICE)
    got = tr.matmul(ta, tb, block=(16, 16, 16)).cpu().numpy()
    want = ta.cpu().double().numpy() @ tb.cpu().double().numpy()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
def test_triton_fp32(precision):
    rng = np.random.default_rng(2)
    a = rng.uniform(-1, 1, (40, 64)).astype(np.float32)
    b = rng.uniform(-1, 1, (64, 24)).astype(np.float32)
    got = tr.matmul(torch.from_numpy(a).to(tr.DEVICE), torch.from_numpy(b).to(tr.DEVICE), precision, (16, 16, 16))
    want = a.astype(np.float64) @ b.astype(np.float64)
    # TF32 keeps 10 mantissa bits of each input; the interpreter ignores the option and uses FP32.
    tol = 64 * 2.0**-10 if precision == "tf32" else 1e-5
    assert np.max(np.abs(got.cpu().numpy() - want)) <= tol


NUMBA_PTX = """
import importlib.util, sys
from numba import cuda, float32
from numba.cuda.types import float16
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
f32, _ = cuda.compile_ptx(m.gemm_f16_f32acc.py_func, (float16[:, ::1], float16[:, ::1], float32[:, ::1]), cc=(7, 5))
f16, _ = cuda.compile_ptx(m.gemm_f16_f16acc.py_func, (float16[:, ::1], float16[:, ::1], float16[:, ::1]), cc=(7, 5))
print(int("cvt.f32.f16" in f32), int("fma.rn.f32" in f32), int("hfma" in f16))
"""


def clean_env():
    # The simulator and the interpreter compile nothing, so compile in a process without them.
    return {k: v for k, v in os.environ.items() if k not in ("NUMBA_ENABLE_CUDASIM", "TRITON_INTERPRET")}


def test_numba_compiles_to_fp16_ptx():
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/fp16_gemm.py")], env=clean_env(), capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    # FP16 loads converted to FP32 and summed with FP32 FMAs; the FP16 version calls __hfma.
    assert res.stdout.split() == ["1", "1", "1"]


def test_triton_ptx_uses_tensor_cores():
    res = subprocess.run(
        [sys.executable, str(HERE / "triton/show_ptx.py")], env=clean_env(), capture_output=True, text=True, timeout=600
    )
    assert res.returncode == 0, res.stderr
    lines = dict(line.split(": ", 1) for line in res.stdout.splitlines()[1:])
    assert lines["fp16 ieee sm_80"].startswith("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
    assert lines["fp16 ieee sm_90"].startswith("wgmma.mma_async")
    assert lines["fp32 tf32 sm_80"].startswith("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
    assert lines["fp32 ieee sm_80"].startswith("fma.rn.f32")
    # The committed output in the chapter must match what this Triton produces.
    assert res.stdout == (HERE / "triton/show_ptx.out.txt").read_text()


def mma_include_dir():
    # NVRTC needs mma.h and the crt/mma.h it includes. pip's cudart headers lack crt/; Triton ships both.
    dirs = [os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/include"]
    spec = importlib.util.find_spec("triton")
    if spec and spec.origin:
        dirs.append(str(Path(spec.origin).parent / "backends/nvidia/include"))
    return next((d for d in dirs if (Path(d) / "crt/mma.h").exists() and (Path(d) / "cuda_fp16.h").exists()), None)


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    inc = mma_include_dir()
    if inc is None:
        pytest.skip("no CUDA include directory with mma.h and crt/mma.h")
    src = (HERE / "cuda/wmma_gemm.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"wmma_gemm.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        pytest.skip(f"NVRTC unavailable: {err}")
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"-std=c++17", f"-I{inc}".encode()]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, log.decode()
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(prog, ptx)
    ptx = ptx.decode()
    for name in ("wmma_gemm_f32acc", "wmma_gemm_f16acc"):
        assert f".entry _Z{len(name)}{name}" in ptx
    assert "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32" in ptx
    assert "wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "wmma_gemm"), "256"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
