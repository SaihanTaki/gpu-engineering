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
    # numba/row_sums.py and triton/row_sums.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/row_sums.py", "ch08_numba_row_sums")
tr = load("triton/row_sums.py", "ch08_triton_row_sums")

# One row, a ragged last block, and several column tiles.
SHAPES = [(1, 32), (45, 64), (100, 96), (4096, 256)]
# The Numba simulator runs one Python thread per GPU thread, so the largest shape is too slow there.
NUMBA_SHAPES = SHAPES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SHAPES


def matrix(rows, cols):
    # Small integers keep every float sum exact, so any summation order matches.
    rng = np.random.default_rng(rows * 1000 + cols)
    return rng.integers(-8, 9, size=(rows, cols)).astype(np.float32)


@pytest.mark.parametrize("shape", NUMBA_SHAPES)
@pytest.mark.parametrize("kernel", ["row_sums_plain", "row_sums_padded", "row_sums_swizzled"])
def test_numba(kernel, shape):
    a = matrix(*shape)
    np.testing.assert_array_equal(nb.run(getattr(nb, kernel), a), a.sum(axis=1))


def test_numba_rejects_ragged_columns():
    with pytest.raises(ValueError):
        nb.run(nb.row_sums_plain, matrix(4, 33))


@pytest.mark.parametrize("shape", SHAPES + [(7, 33)])
def test_triton(shape):
    a = matrix(*shape)
    out = tr.row_sums(torch.from_numpy(a).to(tr.DEVICE))
    np.testing.assert_array_equal(out.cpu().numpy(), a.sum(axis=1))


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("shape", SHAPES)
def test_cuda_binary(shape):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(HERE / "cuda" / "row_sums"), *map(str, shape)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
