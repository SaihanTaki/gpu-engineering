import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/memory_spaces.py", "ch09_numba_memory_spaces")
tr = load("triton/cache_hints.py", "ch09_triton_cache_hints")


@pytest.mark.parametrize("n", [1, 255, 1000])
def test_numba_poly_constant(n):
    x = np.random.default_rng(n).uniform(-1, 1, n).astype(np.float32)
    np.testing.assert_allclose(nb.run_poly(x), nb.poly_reference(x), rtol=1e-5)


@pytest.mark.parametrize("threads", [1, 300])
def test_numba_bin_counts(threads):
    keys = np.random.default_rng(threads).integers(0, 1 << 20, threads * nb.KEYS_PER_THREAD, dtype=np.int32)
    counts = nb.run_bins(keys)
    assert np.array_equal(counts, nb.bins_reference(keys))
    assert counts.sum() == keys.size


@pytest.mark.parametrize("n", [5, 1024, 3000])
@pytest.mark.parametrize("variant", list(tr.VARIANTS))
def test_triton(n, variant):
    x = torch.randn(n, device=tr.DEVICE)
    w = torch.randn(tr.TABLE, device=tr.DEVICE)
    expected = x * w[torch.arange(n, device=tr.DEVICE) % tr.TABLE]
    torch.testing.assert_close(tr.run(x, w, variant), expected)


def ptxas_report():
    """{kernel: (registers, stack bytes, spill store bytes, spill load bytes)} from generated/ptxas-v.txt."""
    text = (HERE / "generated" / "ptxas-v.txt").read_text()
    out = {}
    pattern = (r"Function properties for _Z(\d+)(\w+)\n.*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
               r"(\d+) bytes spill loads\n.*?Used (\d+) registers")
    for length, mangled, stack, stores, loads, regs in re.findall(pattern, text):
        # Itanium mangling: _Z, the name's length, the name, then the parameter types.
        out[mangled[: int(length)]] = (int(regs), int(stack), int(stores), int(loads))
    return out


def test_generated_report_shows_the_spill():
    r = ptxas_report()
    assert set(r) == {"poly_constant", "poly_readonly", "bin_counts", "mirror", "mirror_capped"}
    assert r["mirror_capped"][0] == 32 and r["mirror_capped"][2] > 0
    assert r["mirror"][2:] == (0, 0)
    # A local array without spills: stack frame, but no spill traffic.
    assert r["bin_counts"][1] == 16 * 4 and r["bin_counts"][2:] == (0, 0)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
def test_cuda_binary():
    result = subprocess.run(["make", "-C", str(HERE / "cuda"), "run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
