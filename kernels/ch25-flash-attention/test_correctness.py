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


tr = load("triton/flash_attention.py", "ch25_triton_flash_attention")


def reference(q, k, v, causal):
    # softmax(q k^T / sqrt(d)) v in float64, with the row max subtracted for stability.
    n, d = q.shape[-2], q.shape[-1]
    s = q.astype(np.float64) @ np.swapaxes(k, -1, -2).astype(np.float64) / np.sqrt(d)
    if causal:
        s = np.where(np.triu(np.ones((n, n), dtype=bool), 1), -np.inf, s)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)) @ v.astype(np.float64)


def inputs(b, h, n, d, seed=0):
    rng = np.random.default_rng(seed + n * 100 + d)
    return [rng.standard_normal((b, h, n, d), dtype=np.float32) for _ in range(3)]


def run_triton(q, k, v, causal, dtype, block_m=16, block_n=16):
    t = [torch.from_numpy(x).to(tr.DEVICE, dtype) for x in (q, k, v)]
    return tr.flash_attention(*t, causal=causal, block_m=block_m, block_n=block_n).float().cpu().numpy()


# One row, a partial block, exact blocks, and several blocks with a ragged end.
SIZES = [1, 7, 16, 33, 64]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("d", [16, 32])
def test_triton_fp32(n, d, causal):
    q, k, v = inputs(2, 2, n, d)
    np.testing.assert_allclose(run_triton(q, k, v, causal, torch.float32), reference(q, k, v, causal), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [33, 64])
def test_triton_fp16(n, causal):
    q, k, v = inputs(1, 3, n, 32)
    # P is rounded to FP16 before the second tl.dot, as in the GPU kernel; the accumulator stays FP32.
    q16, k16, v16 = (x.astype(np.float16).astype(np.float32) for x in (q, k, v))
    got = run_triton(q, k, v, causal, torch.float16)
    np.testing.assert_allclose(got, reference(q16, k16, v16, causal), rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("block_m,block_n", [(16, 32), (32, 16)])
def test_triton_unequal_blocks(block_m, block_n):
    # With Br != Bc the causal skip and the diagonal mask meet at different tile edges.
    q, k, v = inputs(1, 1, 50, 16)
    for causal in (False, True):
        got = run_triton(q, k, v, causal, torch.float32, block_m, block_n)
        np.testing.assert_allclose(got, reference(q, k, v, causal), rtol=1e-4, atol=1e-5)


def test_causal_first_row_is_v0():
    # Row 0 may only attend to key 0, so its output is exactly v[0].
    q, k, v = inputs(1, 1, 20, 16)
    np.testing.assert_allclose(run_triton(q, k, v, True, torch.float32)[0, 0, 0], v[0, 0, 0], rtol=1e-6)


@pytest.mark.parametrize("causal", [False, True])
def test_naive_matches_numpy(causal):
    q, k, v = inputs(2, 2, 24, 16)
    # naive_attention takes the softmax in FP32, like PyTorch code that upcasts FP16 scores.
    got = tr.naive_attention(*(torch.from_numpy(x).double() for x in (q, k, v)), causal=causal).numpy()
    np.testing.assert_allclose(got, reference(q, k, v, causal), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("arch", ["75", "80", "90"])
def test_cuda_kernel_compiles_nvrtc(arch):
    nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
    src = (HERE / "cuda/flash_attention_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"flash_attention_kernels.cu", 0, [], [])
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
    assert ".visible .entry flash_attention_fwd" in ptx
    # K and V tiles go through shared memory; plain FP32 FMAs, no tensor core instructions.
    assert "st.shared.f32" in ptx and "ld.shared" in ptx
    assert not re.search(r"\b(w?mma|wgmma)\.", ptx)
    # S is never stored: the only global stores are the output rows.
    assert ptx.count("st.global") == 64
    # With d fixed and the loops unrolled, q, acc and s stay in registers, not local memory.
    assert ".local" not in ptx


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("n,causal", [(1, 0), (100, 0), (100, 1), (1024, 1)])
def test_cuda_binary(n, causal):
    build = subprocess.run(["make", "-C", str(HERE / "cuda")], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(HERE / "cuda" / "flash_attention"), str(n), str(causal)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
