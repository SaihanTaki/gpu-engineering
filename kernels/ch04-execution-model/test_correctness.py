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
    # numba/divergence.py and triton/divergence.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/divergence.py", "ch04_numba_divergence")
tr = load("triton/divergence.py", "ch04_triton_divergence")

# One partial warp, a ragged last warp and block, and a large exact multiple.
SIZES = [1, 31, 1000, 1 << 20]
# The Numba simulator runs one Python thread per GPU thread, so 1 << 20 is too slow for a test there.
NUMBA_SIZES = SIZES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SIZES


def inputs(n, order):
    x = np.random.default_rng(n).standard_normal(n, dtype=np.float32)
    # Sorted inputs make almost every warp take one path; random signs split nearly every warp.
    return np.sort(x) if order == "sorted" else x


def reference(x):
    return np.where(x > 0, x * np.float32(2) + np.float32(1), x * np.float32(0.5))


@pytest.mark.parametrize("order", ["random", "sorted"])
@pytest.mark.parametrize("n", NUMBA_SIZES)
@pytest.mark.parametrize("kernel", ["branchy", "branch_free"])
def test_numba(kernel, n, order):
    x = inputs(n, order)
    np.testing.assert_array_equal(nb.run(getattr(nb, kernel), x), reference(x))


@pytest.mark.parametrize("order", ["random", "sorted"])
@pytest.mark.parametrize("n", SIZES)
def test_triton(n, order):
    x = inputs(n, order)
    out = tr.select(torch.from_numpy(x).to(tr.DEVICE))
    np.testing.assert_array_equal(out.cpu().numpy(), reference(x))


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", SIZES)
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(HERE / "cuda" / "divergence"), str(n)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
