import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def load(path, name):
    # numba/matmul.py and triton/matmul.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/matmul.py", "ch07_numba_matmul")
tr = load("triton/matmul.py", "ch07_triton_matmul")

# One element, a partial tile, an exact multiple of 16 and 32, and a ragged size.
SIZES = [1, 7, 32, 50, 256]
# The Numba simulator runs one Python thread per GPU thread, so large n is too slow there.
NUMBA_SIZES = SIZES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SIZES


def inputs(n):
    rng = np.random.default_rng(n)
    A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    B = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    return A, B


def check(C, A, B):
    n = A.shape[0]
    # float32 sums in a different order than the float64 reference.
    np.testing.assert_allclose(C, A.astype(np.float64) @ B, rtol=1e-4, atol=1e-5 * n)


@pytest.mark.parametrize("n", NUMBA_SIZES)
@pytest.mark.parametrize("kernel", ["matmul_naive", "matmul_tiled"])
def test_numba(kernel, n):
    A, B = inputs(n)
    check(nb.run(getattr(nb, kernel), A, B), A, B)


@pytest.mark.parametrize("n", SIZES)
def test_triton(n):
    A, B = inputs(n)
    C = tr.matmul(torch.from_numpy(A).to(tr.DEVICE), torch.from_numpy(B).to(tr.DEVICE))
    check(C.cpu().numpy(), A, B)


def test_generated_files_exist():
    for name in ("numba_tiled.ptx", "triton.ttgir", "shared_ops.txt", "VERSIONS.txt"):
        assert (HERE / "generated" / name).stat().st_size > 0


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 50, 512])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "matmul"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
