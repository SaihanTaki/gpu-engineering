import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/unified_memory.py", "ch10_numba_unified_memory")

# Smaller than one block, a ragged last block, and an exact multiple of 256.
SIZES = [1, 1000, 4096]


def data(n):
    return np.random.default_rng(n).standard_normal(n, dtype=np.float32)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mode", ["pageable", "pinned", "mapped"])
def test_numba(mode, n):
    x = data(n)
    np.testing.assert_array_equal(nb.RUNS[mode](x), 2 * x)


@pytest.mark.skipif(SIM, reason="the Numba CUDA simulator has no managed_array")
@pytest.mark.parametrize("n", SIZES)
def test_numba_managed(n):
    x = data(n)
    np.testing.assert_array_equal(nb.run_managed(x), 2 * x)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n", SIZES + [1 << 22])
def test_cuda_binary(n):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "unified_memory"), str(n)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
