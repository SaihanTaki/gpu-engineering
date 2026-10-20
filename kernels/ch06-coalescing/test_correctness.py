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
    # numba/coalescing.py and triton/coalescing.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/coalescing.py", "ch06_numba_coalescing")
tr = load("triton/coalescing.py", "ch06_triton_coalescing")

# One partial warp, a ragged last block, and a large exact multiple.
SIZES = [1, 1000, 1 << 18]
# The Numba simulator runs one Python thread per GPU thread, so 1 << 18 is too slow for a test there.
NUMBA_SIZES = SIZES[:-1] if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else SIZES
OFFSETS = [0, 1, 8, 31]
STRIDES = [1, 2, 3, 8, 32]


def source(n):
    return np.arange(n * 32 + 32, dtype=np.float32)


def particles(n):
    return np.random.default_rng(n).standard_normal((n, 3), dtype=np.float32)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba_offset(n, offset):
    src = source(n)
    np.testing.assert_array_equal(nb.run(nb.copy_offset, src, n, offset), src[offset : offset + n])


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba_strided(n, stride):
    src = source(n)
    np.testing.assert_array_equal(nb.run(nb.copy_strided, src, n, stride), src[: n * stride : stride])


@pytest.mark.parametrize("n", NUMBA_SIZES)
def test_numba_aos_soa(n):
    p = particles(n)
    np.testing.assert_array_equal(nb.run(nb.get_x_aos, p, n), p[:, 0])
    x = np.ascontiguousarray(p[:, 0])
    np.testing.assert_array_equal(nb.run(nb.get_x_soa, x, n), p[:, 0])


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("n", SIZES)
def test_triton_strided(n, stride, offset):
    src = source(n)
    out = tr.copy_strided(torch.from_numpy(src).to(tr.DEVICE), n, stride, offset)
    np.testing.assert_array_equal(out.cpu().numpy(), src[offset : offset + n * stride : stride])


@pytest.mark.parametrize("n", SIZES)
def test_triton_aos(n):
    p = particles(n)
    out = tr.get_x_aos(torch.from_numpy(p).to(tr.DEVICE))
    np.testing.assert_array_equal(out.cpu().numpy(), p[:, 0])


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", [1, 1000, 1 << 20])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "coalescing"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
