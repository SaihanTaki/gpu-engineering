import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def load(path, name):
    # numba/bugs.py and triton/bugs.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/bugs.py", "ch22_numba_bugs")
tr = load("triton/bugs.py", "ch22_triton_bugs")

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
# One element, a partial block, and several blocks.
SIZES = [1, 300, 1000]


@pytest.mark.parametrize("n", SIZES)
def test_numba_atomic_count(n):
    x = nb.make_input(n)
    assert nb.count("atomic", x) == int((x > 0).sum())


@pytest.mark.skipif(SIM, reason="the simulator almost never interleaves inside count[0] += 1")
def test_numba_racy_count_loses_updates():
    # On a GPU, the 32 lanes of a warp load the counter together, so updates are lost.
    x = np.ones(1 << 16, dtype=np.float32)
    assert nb.count("racy", x) < x.size


@pytest.mark.parametrize("n", SIZES)
def test_numba_scale(n):
    x = nb.make_input(n)
    np.testing.assert_array_equal(nb.run(nb.scale, x), 2 * x)


@pytest.mark.skipif(not SIM, reason="only the simulator checks bounds")
def test_numba_simulator_catches_oob():
    with pytest.raises(IndexError, match="out of bounds"):
        nb.run(nb.scale_unguarded, nb.make_input(300))


@pytest.mark.skipif(not SIM, reason="shows a simulator blind spot")
def test_numba_simulator_wraps_negative_index():
    x = np.arange(5, dtype=np.float32)
    out = nb.run(nb.diff_wrap, x)
    # out[0] = x[0] - x[-1]: no error, just a wrong value.
    assert out[0] == x[0] - x[4]
    np.testing.assert_array_equal(out[1:], np.diff(x))


@pytest.mark.parametrize("n", SIZES + [tr.BLOCK, 3 * tr.BLOCK])
def test_triton_masked(n):
    x = torch.arange(n, dtype=torch.float32)
    out, intact = tr.scale("masked", x)
    assert torch.equal(out, 2 * x)
    assert intact


@pytest.mark.parametrize("n", [1, 300, 1000])
def test_triton_unmasked_writes_past_end(n):
    # A size that is not a multiple of BLOCK: the last program stores into the guard band.
    out, intact = tr.scale("unmasked", torch.arange(n, dtype=torch.float32))
    assert torch.equal(out, 2 * torch.arange(n, dtype=torch.float32))
    assert not intact


def test_triton_unmasked_hides_on_round_sizes():
    # With n a multiple of BLOCK every lane is in bounds, so a test at this size passes.
    n = 4 * tr.BLOCK
    out, intact = tr.scale("unmasked", torch.arange(n, dtype=torch.float32))
    assert torch.equal(out, 2 * torch.arange(n, dtype=torch.float32))
    assert intact


NUMBA_PTX = """
from numba import cuda, float32, uint32
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for name, k in m.COUNTERS.items():
    ptx, _ = cuda.compile_ptx(k.py_func, (float32[::1], uint32[::1]), cc=(8, 0))
    print(name, ptx.count("atom.shared"), ptx.count("atom.global"))
"""


def test_numba_compiles_for_gpu():
    # The simulator compiles nothing, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/bugs.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    # The racy counter has no shared atomic; the fixed one has one.
    assert counts["racy"][0] == 0 and counts["racy"][1] > 0
    assert counts["atomic"][0] > 0


def nvrtc_ptx(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/bugs_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"bugs_kernels.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        pytest.skip(f"NVRTC unavailable: {err}")
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, log.decode()
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(prog, ptx)
    return ptx.decode()


def body(ptx, kernel):
    """PTX of one kernel, found by its mangled name."""
    m = re.search(r"\.entry (_Z\d+" + kernel + r"P\w+)\(.*?\n}\n", ptx, re.S)
    assert m, kernel
    return m.group(0)


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    ptx = nvrtc_ptx(arch)
    racy, atomic = body(ptx, "count_racy"), body(ptx, "count_atomic")
    # count += 1 on shared memory is a plain load, add and store.
    assert re.search(r"ld\.shared\.u32[^\n]*\n\s*add\.s32[^\n]*\n\s*st\.shared\.u32", racy)
    assert "atom.shared" not in racy
    assert "atom.shared.add.u32" in atomic
    # The missing barrier is missing in the PTX too.
    assert "bar.sync" not in body(ptx, "reverse_nosync")
    assert body(ptx, "reverse").count("bar.sync") == 1
    # One barrier in each branch of the divergent kernel, one shared barrier in the fix.
    assert body(ptx, "block_sum_divergent").count("bar.sync") == 2
    assert body(ptx, "block_sum").count("bar.sync") == 1


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("bug", ["race", "oob", "nosync", "uninit", "divergent", "launch"])
def test_cuda_binary(bug):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "bugs"), bug, "100000"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(rf"{bug}\s+fixed\s+OK", result.stdout)
