import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load(path, name):
    # numba/whoami.py and triton/whoami.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/whoami.py", "ch02_numba_whoami")
tr = load("triton/whoami.py", "ch02_triton_whoami")

# (n, block size): exact multiple, ragged last block, smaller than one block.
CASES_1D = [(32, 8), (20, 8), (5, 8)]
CASES_2D = [(8, 4), (5, 3), (1, 1)]


@pytest.mark.parametrize("n, threads", CASES_1D)
def test_numba_1d(n, threads):
    out, block, thread = nb.run(n, threads)
    i = np.arange(n)
    np.testing.assert_array_equal(out, i)
    np.testing.assert_array_equal(block, i // threads)
    np.testing.assert_array_equal(thread, i % threads)


@pytest.mark.parametrize("width, height", CASES_2D)
def test_numba_2d(width, height):
    out = nb.run2d(width, height)
    np.testing.assert_array_equal(out, np.arange(width * height).reshape(height, width))


@pytest.mark.parametrize("n, block", CASES_1D)
def test_triton_1d(n, block):
    out, blocks = tr.run(n, block)
    i = np.arange(n)
    np.testing.assert_array_equal(out, i)
    np.testing.assert_array_equal(blocks, i // block)


@pytest.mark.parametrize("width, height", CASES_2D)
def test_triton_2d(width, height):
    out = tr.run2d(width, height)
    np.testing.assert_array_equal(out, np.arange(width * height).reshape(height, width))


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    result = subprocess.run(
        ["make", "-C", str(HERE / "cuda"), "run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
