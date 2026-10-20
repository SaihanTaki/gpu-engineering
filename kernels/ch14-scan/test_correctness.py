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
    # numba/scan.py and triton/scan.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/scan.py", "ch14_numba_scan")
tr = load("triton/scan.py", "ch14_triton_scan")

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
# One element, exact tiles of 256 and 512, one past them, and several tiles of each.
SIZES = [1, 7, 256, 257, 512, 513, 1000, 5000]
# More than 512 tiles of 512 (Blelloch) needs a second level of recursion; the simulator is too slow for it.
NUMBA_SIZES = SIZES if SIM else SIZES + [70_000, 300_000]
# With 16 elements per program, 5000 elements recurse three levels deep: 313, 20 and 2 tiles.
TRITON_BLOCKS = [tr.BLOCK, 16]


def data(n):
    return np.random.default_rng(n).integers(0, 10, n, dtype=np.int32)


def reference(x, inclusive):
    incl = np.cumsum(x, dtype=np.int64)
    return incl if inclusive else incl - x


@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("algo", list(nb.ALGOS))
@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba(algo, inclusive, n):
    x = data(n)
    np.testing.assert_array_equal(nb.scan(x, algo, inclusive), reference(x, inclusive))


@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("variant", list(tr.VARIANTS))
@pytest.mark.parametrize("block", TRITON_BLOCKS)
@pytest.mark.parametrize("n", SIZES)
def test_triton(variant, block, inclusive, n):
    x = data(n)
    out = tr.scan(torch.from_numpy(x).to(tr.DEVICE), inclusive, variant, block)
    np.testing.assert_array_equal(out.cpu().numpy(), reference(x, inclusive))


NUMBA_PTX = """
from numba import cuda, int32, boolean
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sig = (int32[::1], int32[::1], int32[::1], boolean)
for name, (k, _) in m.ALGOS.items():
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 0))
    print(name, ptx.count("bar.sync"), ptx.count("st.shared"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run([sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/scan.py")], env=env, capture_output=True, text=True)
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    for line in res.stdout.splitlines():
        name, bars, stores = line.split()
        assert int(bars) > 0 and int(stores) > 0, line


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/scan_kernels.cuh").read_text()
    for k in ("scan_hillis_steele", "scan_blelloch"):
        for b in ("true", "false"):
            src += f"\ntemplate __global__ void {k}<{b}>(const int *, int *, int *, int);"
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"scan_kernels.cu", 0, [], [])
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
    # 256 ints for Hillis-Steele, 512 for Blelloch.
    assert ".b8 _ZZ18scan_hillis_steeleILb1EEvPKiPiS2_iE1s[1024]" in ptx
    assert ".b8 _ZZ13scan_blellochILb1EEvPKiPiS2_iE1s[2048]" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 513, 1_000_000])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "scan"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
