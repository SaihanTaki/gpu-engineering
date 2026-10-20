"""Compile cuda/fusion_kernels.cuh to PTX for sm_80 with NVRTC (no GPU needed) and write generated/."""

import re
from pathlib import Path

from cuda.bindings import nvrtc

HERE = Path(__file__).parent
OUT = HERE / "generated"


def compile_ptx(src, arch="80"):
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"fusion_kernels.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(prog, ptx)
    return ptx.decode().rstrip("\x00")


def main():
    ptx = compile_ptx((HERE / "cuda/fusion_kernels.cuh").read_text())
    OUT.mkdir(exist_ok=True)
    (OUT / "fusion_sm80.ptx").write_text(ptx)
    entries = re.findall(r"(\.visible \.entry (\w+)\(.*?\n\}\n)", ptx, flags=re.S)
    _, major, minor = nvrtc.nvrtcVersion()
    lines = [f"NVRTC {major}.{minor}, compute_80. Per kernel:", ""]
    lines.append(f"{'kernel':<8} {'ld.global':>9} {'st.global':>9} {'fma.rn.f32':>10}")
    for body, mangled in entries:
        name = next(k for k in ("scale", "add", "relu", "fused") if k in mangled)
        counts = [len(re.findall(p, body)) for p in (r"\bld\.global", r"\bst\.global", r"\bfma\.rn\.f32")]
        lines.append(f"{name:<8} {counts[0]:>9} {counts[1]:>9} {counts[2]:>10}")
        if name == "fused":
            (OUT / "fused_sm80.ptx").write_text(body)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
