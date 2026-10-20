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


def load(path, name):
    # numba/gemm.py and triton/gemm.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/gemm.py", "ch20_numba_gemm")
tr = load("triton/gemm.py", "ch20_triton_gemm")

# (M, N, K): one element, smaller than one tile, ragged edges on every side, and several whole tiles.
SHAPES = [(1, 1, 1), (5, 3, 7), (33, 20, 17), (70, 65, 40), (64, 64, 64)]
# The Numba simulator runs one Python thread per GPU thread, so it gets the smaller shapes.
NUMBA_SHAPES = SHAPES[:3] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SHAPES


def operands(shape):
    m, n, k = shape
    rng = np.random.default_rng(m * 10000 + n * 100 + k)
    # Whole numbers from -4 to 4: every partial sum is exact in float32, in any order.
    a = rng.integers(-4, 5, (m, k)).astype(np.float32)
    b = rng.integers(-4, 5, (k, n)).astype(np.float32)
    return a, b


@pytest.mark.parametrize("name", list(nb.KERNELS))
@pytest.mark.parametrize("shape", NUMBA_SHAPES)
def test_numba(name, shape):
    a, b = operands(shape)
    np.testing.assert_array_equal(nb.run(name, a, b), a @ b)


@pytest.mark.parametrize("shape", SHAPES)
def test_triton(shape):
    a, b = operands(shape)
    out = tr.gemm(torch.from_numpy(a).to(tr.DEVICE), torch.from_numpy(b).to(tr.DEVICE))
    np.testing.assert_array_equal(out.cpu().numpy(), a @ b)


NUMBA_PTX = """
from numba import cuda, float32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sig = (float32[:, ::1], float32[:, ::1], float32[:, ::1])
for name, k in m.KERNELS.items():
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 0))
    print(name, ptx.count("ld.shared"), ptx.count("bar.sync"), ptx.count("local"))
"""


def test_numba_compiles_for_gpu():
    # The simulator compiles nothing, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/gemm.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    assert counts["naive"][:2] == (0, 0)
    for name in ("tiled", "regblock"):
        assert counts[name][0] > 0 and counts[name][1] > 0, counts


INSTANCES = """
template __global__ void gemm_smem<32, 32, 32>(const float *, const float *, float *, int, int, int);
template __global__ void gemm_1d<64, 64, 8, 8>(const float *, const float *, float *, int, int, int);
template __global__ void gemm_2d<128, 128, 8, 8, 8>(const float *, const float *, float *, int, int, int);
template __global__ void gemm_vec<128, 128, 8, 8, 8>(const float *, const float *, float *, int, int, int);
"""


def nvrtc_ptx(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/gemm_kernels.cuh").read_text() + INSTANCES
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"gemm_kernels.cu", 0, [], [])
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
    return ptx.decode()


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    ptx = nvrtc_ptx(arch)
    parts = {re.search(r"\.entry (\w+)", p).group(1): p for p in re.split(r"(?=\.(?:visible|weak) \.entry )", ptx) if ".entry" in p}
    vec = next(p for name, p in parts.items() if "gemm_vec" in name)
    # The float4 kernel loads 16 bytes per instruction from global and shared memory, and nothing narrower.
    assert re.search(r"ld\.global(\.nc)?\.v4\.f32", vec) and "ld.shared.v4.f32" in vec
    assert not re.search(r"ld\.(global|shared)(\.nc)?\.f32", vec)
    # No kernel spills its accumulators to local memory.
    assert ".local" not in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("shape", [(1, 1, 1), (33, 20, 17), (70, 68, 40), (256, 256, 256)])
def test_cuda_binary(shape):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "gemm"), *map(str, shape)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
