import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/streams.py", "ch21_numba_streams")

ITERS = 7


def data(n):
    return np.random.default_rng(n).standard_normal(n, dtype=np.float32)


@pytest.mark.parametrize("n", [1, 1000, 4096])
def test_serial_matches_numpy(n):
    x = data(n)
    # The GPU may fuse v * a + b into one FMA, so allow a few ulps.
    np.testing.assert_allclose(nb.serial(x, ITERS), nb.reference(x, ITERS), rtol=1e-5, atol=1e-6)


# One chunk, chunks that divide n, a ragged last chunk, more chunks than streams, and more chunks than elements.
@pytest.mark.parametrize("n,streams,chunks", [(1000, 1, 1), (1024, 2, 4), (1000, 3, 7), (4096, 4, 16), (3, 2, 8)])
def test_pipeline_matches_serial(n, streams, chunks):
    x = data(n)
    np.testing.assert_array_equal(nb.streamed(x, streams, chunks, ITERS), nb.serial(x, ITERS))


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernel_compiles_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/streams_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"streams_kernels.cu", 0, [], [])
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
    assert ".visible .entry _Z4workPfii" in ptx
    # The _rn intrinsics keep the multiply and the add apart: no fma in the loop.
    assert "mul.rn.f32" in ptx and "add.rn.f32" in ptx and "fma" not in ptx


NUMBA_PTX = """
import importlib.util, sys
from numba import cuda, float32, int32
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ptx, _ = cuda.compile_ptx(m.work.py_func, (float32[::1], int32), cc=(8, 0))
print(ptx.count(".entry"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/streams.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("args", [["1"], ["1000", "3", "7", "5"], [str(1 << 20), "4", "16", "50"]])
def test_cuda_binary(args):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "streams"), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
