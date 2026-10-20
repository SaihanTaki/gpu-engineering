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


nb = load("numba/saxpy.py", "ch0_2_numba_saxpy")
tr = load("triton/saxpy.py", "ch0_2_triton_saxpy")

# Exact multiple of the block, ragged last block, smaller than one block.
SIZES = [2048, 1000, 5]


def inputs(n):
    rng = np.random.default_rng(n)
    return rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)


@pytest.mark.parametrize("n", SIZES)
def test_numba(n):
    x, y = inputs(n)
    np.testing.assert_allclose(nb.run(2.5, x, y), 2.5 * x + y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", SIZES)
def test_triton(n):
    x, y = inputs(n)
    np.testing.assert_allclose(tr.run(2.5, x, y), 2.5 * x + y, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    result = subprocess.run(
        ["make", "-C", str(HERE / "cuda"), "run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


GENERATED = [
    "VERSIONS.txt",
    "numba.ll",
    "numba.ptx",
    "numba.cubin",
    "numba.sass",
    "triton.ttir",
    "triton.ttgir",
    "triton.llir",
    "triton.ptx",
    "triton.cubin",
    "triton.sass",
]


@pytest.mark.parametrize("name", GENERATED)
def test_generated_listing_present(name):
    # Checks the committed output of generate.py; it does not regenerate it.
    path = HERE / "generated" / name
    assert path.is_file()
    assert path.stat().st_size > 0
