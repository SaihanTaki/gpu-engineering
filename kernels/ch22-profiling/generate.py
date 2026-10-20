"""Compiles cuda/bugs_kernels.cuh with NVRTC for sm_80 (no GPU needed) and
writes the PTX of the racy and atomic counters to generated/."""

import re
from pathlib import Path

from cuda.bindings import nvrtc

HERE = Path(__file__).parent


def ptx_for(arch="80"):
    src = (HERE / "cuda/bugs_kernels.cuh").read_text()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"bugs_kernels.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(prog, ptx)
    return ptx.decode().rstrip("\x00")


def main():
    ptx = ptx_for()
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    for kernel in ("count_racy", "count_atomic"):
        body = re.search(r"\.visible \.entry _Z\d+" + kernel + r"P\w+\(.*?\n}\n", ptx, re.S).group(0)
        (out / f"{kernel}.ptx").write_text(f"// NVRTC, compute_80, from cuda/bugs_kernels.cuh\n{body}")
        print(f"wrote generated/{kernel}.ptx")


if __name__ == "__main__":
    main()
