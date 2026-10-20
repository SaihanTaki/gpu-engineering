import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def load(path, name):
    # numba/spmv.py and triton/spmv.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if SIM:
    from numba.cuda.simulator.kernelapi import FakeCUDAModule

    def shfl_down_sync(self, mask, value, delta):
        # The simulator has no warp shuffles. This one uses block barriers, so every
        # thread of the block must call it equally often, which spmv_csr_vector does.
        t = threading.current_thread()
        slots = t._manager.__dict__.setdefault("shfl_slots", {})
        slots[t.thread_id] = value
        t.syncthreads()
        src = t.thread_id + delta
        same_warp = src // 32 == t.thread_id // 32 and src in slots
        out = slots[src] if same_warp else value
        t.syncthreads()
        return out

    FakeCUDAModule.shfl_down_sync = shfl_down_sync

nb = load("numba/spmv.py", "ch18_numba_spmv")
tr = load("triton/spmv.py", "ch18_triton_spmv")


def matrix(kind, rows, cols, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((rows, cols), dtype=np.float32)
    if kind == "banded":
        for r in range(rows):
            for c in range(max(0, r - 1), min(cols, r + 2)):
                a[r, c] = rng.uniform(-1, 1)
    elif kind == "random":
        mask = rng.random((rows, cols)) < 0.2
        a[mask] = rng.uniform(-1, 1, mask.sum())
    elif kind == "power-law":
        # Row r has about cols / (r + 1) nonzeros, so row 0 is full and longer than a warp.
        for r in range(rows):
            n = min(cols, -(-cols // (r + 1)))
            a[r, rng.choice(cols, n, replace=False)] = rng.uniform(-1, 1, n)
    elif kind == "empty-rows":
        a[::3, rng.integers(0, cols, (rows + 2) // 3)] = 1.0
    return a


# 1 x 1, ragged sizes that do not fill a block or a warp, rows longer than 32, and empty rows.
CASES = [("banded", 1, 1), ("banded", 37, 37), ("random", 50, 70), ("power-law", 45, 100), ("empty-rows", 33, 20)]


def reference(a, x):
    try:
        import scipy.sparse

        return scipy.sparse.csr_matrix(a) @ x
    except ImportError:
        return a.astype(np.float64) @ x


def inputs(case):
    a = matrix(*case)
    x = np.random.default_rng(1).uniform(-1, 1, a.shape[1]).astype(np.float32)
    return a, x


def test_formats():
    a = np.array([[5, 0, 0, 1], [0, 0, 0, 0], [0, 2, 3, 0]], dtype=np.float32)
    row_ptr, col, val = nb.csr_from_dense(a)
    assert row_ptr.tolist() == [0, 2, 2, 4]
    assert col.tolist() == [0, 3, 1, 2]
    assert val.tolist() == [5, 1, 2, 3]
    ell_col, ell_val = nb.ell_from_csr(row_ptr, col, val)
    # Width 2, column-major: slot k of every row, then slot k + 1.
    assert ell_col.tolist() == [[0, 0, 1], [3, 0, 2]]
    assert ell_val.tolist() == [[5, 0, 2], [1, 0, 3]]


@pytest.mark.parametrize("name", nb.KERNELS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}x{c[2]}")
def test_numba(name, case):
    a, x = inputs(case)
    got = nb.run(name, *nb.csr_from_dense(a), x)
    np.testing.assert_allclose(got, reference(a, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}x{c[2]}")
def test_triton(case):
    a, x = inputs(case)
    row_ptr, col, val = (torch.from_numpy(t).to(tr.DEVICE) for t in nb.csr_from_dense(a))
    got = tr.spmv(row_ptr, col, val, torch.from_numpy(x).to(tr.DEVICE))
    np.testing.assert_allclose(got.cpu().numpy(), reference(a, x), rtol=1e-5, atol=1e-5)


NUMBA_PTX = """
from numba import cuda, float32, int32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
csr = (int32[::1], int32[::1], float32[::1], float32[::1], float32[::1])
ptx, _ = cuda.compile_ptx(m.spmv_csr_vector.py_func, csr, cc=(8, 0))
print(ptx.count("shfl.sync.down"))
"""


def test_numba_vector_compiles_to_shuffles():
    # The simulator compiles nothing, so compile the vector kernel to real PTX in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/spmv.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    # The loop over offsets 16, 8, 4, 2, 1 may be unrolled, so at least one shuffle.
    assert int(res.stdout.strip()) >= 1


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/spmv_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"spmv_kernels.cu", 0, [], [])
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
    for name in ("spmv_csr_scalar", "spmv_csr_vector", "spmv_ell"):
        assert f".entry _Z{len(name)}{name}" in ptx
    assert "shfl.sync.down" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "spmv"), "5000"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
