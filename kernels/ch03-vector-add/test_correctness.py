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
    # numba/vector_add.py and triton/vector_add.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/vector_add.py", "ch03_numba_vector_add")
tr = load("triton/vector_add.py", "ch03_triton_vector_add")

# Smaller than one block, ragged last blocks for 256 and 1024, and a large exact multiple.
SIZES = [1, 1000, 1025, 1 << 20]
# The Numba simulator runs one Python thread per GPU thread, so 1 << 20 is too slow for a test there.
NUMBA_SIZES = SIZES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SIZES


def inputs(n):
    rng = np.random.default_rng(n)
    return rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)


@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba(n):
    a, b = inputs(n)
    np.testing.assert_array_equal(nb.run(a, b), a + b)


@pytest.mark.parametrize("n", SIZES)
def test_triton(n):
    a, b = inputs(n)
    c = tr.vector_add(torch.from_numpy(a).to(tr.DEVICE), torch.from_numpy(b).to(tr.DEVICE))
    np.testing.assert_array_equal(c.cpu().numpy(), a + b)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", SIZES)
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(HERE / "cuda" / "vector_add"), str(n)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
