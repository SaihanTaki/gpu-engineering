import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load(path, name):
    # numba/saxpy.py and triton/saxpy.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/saxpy.py", "ch0_1_numba_saxpy")
tr = load("triton/saxpy.py", "ch0_1_triton_saxpy")

# Smaller than one block, exact multiples, and ragged last blocks for block size 256.
SIZES = [1, 255, 256, 1000, 4097]


def inputs(n):
    rng = np.random.default_rng(n)
    return rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)


def reference(a, x, y):
    return np.float32(a) * x + y


@pytest.mark.parametrize("n", SIZES)
def test_numba_cpu(n):
    x, y = inputs(n)
    np.testing.assert_allclose(nb.run_cpu(2.5, x, y), reference(2.5, x, y), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_numba_gpu(n):
    x, y = inputs(n)
    np.testing.assert_allclose(nb.run(2.5, x, y), reference(2.5, x, y), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_triton(n):
    x, y = inputs(n)
    np.testing.assert_allclose(tr.run(2.5, x, y, block=256), reference(2.5, x, y), rtol=1e-6, atol=1e-6)


def test_inputs_not_modified():
    x, y = inputs(300)
    x0, y0 = x.copy(), y.copy()
    nb.run(2.5, x, y)
    nb.run_cpu(2.5, x, y)
    tr.run(2.5, x, y)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    result = subprocess.run(
        ["make", "-C", str(HERE / "cuda"), "run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
