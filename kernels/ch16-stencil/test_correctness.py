import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def load(path, name):
    # numba/conv2d.py and triton/conv2d.py share a module name, so load by path.
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = load("numba/conv2d.py", "ch16_numba_conv2d")
tr = load("triton/conv2d.py", "ch16_triton_conv2d")


def reference(img, filt):
    """out[y, x] = sum of filt[dy + R, dx + R] * img[y + dy, x + dx], with zeros outside the image."""
    R = filt.shape[0] // 2
    h, w = img.shape
    padded = np.pad(img.astype(np.float64), R)
    out = np.zeros((h, w))
    for dy in range(2 * R + 1):
        for dx in range(2 * R + 1):
            out += filt[dy, dx] * padded[dy : dy + h, dx : dx + w]
    return out


def case(shape, R):
    rng = np.random.default_rng(shape[0] * 100 + shape[1] * 10 + R)
    img = rng.uniform(-1, 1, shape).astype(np.float32)
    filt = rng.uniform(-1, 1, (2 * R + 1, 2 * R + 1)).astype(np.float32)
    return img, filt


# One pixel, one exact tile, ragged tiles, and an image smaller than the halo in one direction.
SHAPES = [(1, 1), (16, 16), (17, 33), (40, 21), (2, 50)]
RADII = [1, 2, 3]


def test_reference_boundary_rule():
    # A box filter of ones on an image of ones counts the in-image neighbours of each pixel.
    out = reference(np.ones((4, 5), np.float32), np.ones((3, 3), np.float32))
    assert out[0, 0] == 4 and out[0, 2] == 6 and out[1, 1] == 9
    # A centred delta returns the image unchanged.
    img, _ = case((6, 7), 2)
    delta = np.zeros((5, 5), np.float32)
    delta[2, 2] = 1
    np.testing.assert_array_equal(reference(img, delta), img)
    # Shifting by one pixel: the weight left of centre picks up the left neighbour.
    left = np.zeros((3, 3), np.float32)
    left[1, 0] = 1
    shifted = reference(img, left)
    np.testing.assert_array_equal(shifted[:, 1:], img[:, :-1])
    assert np.all(shifted[:, 0] == 0)


@pytest.mark.parametrize("name", ["naive", "tiled"])
@pytest.mark.parametrize("R", RADII)
@pytest.mark.parametrize("shape", SHAPES)
def test_numba(name, R, shape):
    img, filt = case(shape, R)
    np.testing.assert_allclose(nb.run(name, img, filt), reference(img, filt), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("R", RADII)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton(R, shape):
    img, filt = case(shape, R)
    out = tr.conv2d(torch.from_numpy(img).to(tr.DEVICE), torch.from_numpy(filt).to(tr.DEVICE))
    np.testing.assert_allclose(out.cpu().numpy(), reference(img, filt), rtol=1e-5, atol=1e-5)


NUMBA_PTX = """
import importlib.util, sys
import numpy as np
from numba import cuda, float32
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sig = (float32[:, ::1], float32[:, ::1])
for name, k in m.make_kernels(np.ones((5, 5), np.float32)).items():
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 0))
    print(name, ptx.count("ld.shared"), ptx.count("st.shared"), ptx.count("bar.sync"))
"""


def test_numba_compiles_for_gpu():
    # The simulator does not compile anything, so compile to PTX for real in a clean process.
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    res = subprocess.run(
        [sys.executable, "-c", NUMBA_PTX, str(HERE / "numba/conv2d.py")], env=env, capture_output=True, text=True
    )
    if res.returncode != 0 and "NVVM" in res.stderr:
        pytest.skip("NVVM not available")
    assert res.returncode == 0, res.stderr
    counts = {line.split()[0]: tuple(map(int, line.split()[1:])) for line in res.stdout.splitlines()}
    assert counts["naive"] == (0, 0, 0)
    assert all(n > 0 for n in counts["tiled"]), counts


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernels_compile_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/conv2d_kernels.cuh").read_text()
    for R in RADII:
        src += f"\ntemplate __global__ void conv2d_naive<{R}>(const float *, float *, int, int);"
        src += f"\ntemplate __global__ void conv2d_tiled<{R}>(const float *, float *, int, int);"
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"conv2d_kernels.cu", 0, [], [])
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
    ptx = ptx.decode()
    # 7 x 7 floats of constant memory, and the R = 1 tile of 18 x 18 floats.
    assert ".const .align 4 .b8 c_filter[196]" in ptx
    assert "_ZZ12conv2d_tiledILi1EEvPKfPfiiE4tile[1296]" in ptx
    assert "ld.const" in ptx and "bar.sync" in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("shape", [(1, 1), (17, 33), (100, 37)])
def test_cuda_binary(shape):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "conv2d"), *map(str, shape)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_generated_summary():
    # generate.py output: the filter reads stay in constant memory in both CUDA kernels.
    rows = {}
    for line in (HERE / "generated/summary.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0].startswith("conv2d_") and len(parts) == 6:
            rows.setdefault(parts[0], []).append(tuple(map(int, parts[1:])))
    naive_cuda, naive_numba = rows["conv2d_naive"]
    tiled_cuda, tiled_numba = rows["conv2d_tiled"]
    # Columns: ld.const, ld.global, ld.shared, st.shared, bar.sync. R = 2 gives 25 weights.
    assert naive_cuda[0] == 25 and naive_cuda[2:] == (0, 0, 0)
    assert tiled_cuda[0] == 25 and tiled_cuda[2] == 25 and tiled_cuda[4] == 1
    assert tiled_numba[2] == 25 and tiled_numba[4] == 1
    assert rows["conv2d_kernel"][0][2:] == (0, 0, 0)
