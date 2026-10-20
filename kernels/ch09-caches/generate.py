"""Compile the chapter's kernels for sm_80 and save what the compilers report
to generated/. Needs no GPU, no driver and no nvcc.

- ptxas-v.txt: cuda/memory_spaces.cu compiled by NVRTC with -Xptxas -v, the
  same ptxas report `nvcc -Xptxas -v` prints. Only the "Compile time" lines
  are dropped, because they change on every run.
- memory-ops.txt: per kernel, how many PTX instructions of each memory kind
  the CUDA C++ (NVRTC) and Numba (NVVM) compilers emit.
- triton-loads.txt: the PTX load instructions Triton emits for each set of
  cache hints in triton/cache_hints.py.

    python kernels/ch09-caches/generate.py

Rerun after changing a kernel or a pinned version, and commit the results.
"""

import os
import platform
import re
import sys
import tempfile
import types
from collections import Counter
from importlib import metadata
from pathlib import Path

# These switch Numba and Triton to CPU modes that cannot emit GPU code.
os.environ.pop("NUMBA_ENABLE_CUDASIM", None)
os.environ.pop("TRITON_INTERPRET", None)
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-")

import numpy as np
import triton
from cuda.bindings import nvrtc
from numba import cuda, float32, int32
from triton.backends.compiler import GPUTarget

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
CC = (8, 0)
# Columns of memory-ops.txt. "ld" and "st" are generic loads and stores,
# whose address can point to any state space.
KINDS = ["ld.const", "ld.global.nc", "ld.global", "ld", "ld.local", "st.local", "st.global", "st"]
SPACES = {"param", "const", "global", "local", "shared"}


def load(relpath, name):
    module = types.ModuleType(name)
    module.__file__ = relpath
    sys.modules[name] = module
    exec(compile(Path(relpath).read_text(), relpath, "exec"), module.__dict__)
    return module


def write(name, text):
    (OUT / name).write_text(text.rstrip("\n") + "\n")
    print(f"wrote generated/{name}")


def check(result):
    err = result[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error {err}")
    return result[1:]


def nvrtc_compile(src, opts):
    (prog,) = check(nvrtc.nvrtcCreateProgram(src.encode(), b"memory_spaces.cu", 0, [], []))
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])[0]
    (size,) = check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * size
    check(nvrtc.nvrtcGetProgramLog(prog, log))
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(log.decode())
    return prog, log.decode().rstrip("\0 \n")


def nvrtc_ptx(prog):
    (size,) = check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * size
    check(nvrtc.nvrtcGetPTX(prog, ptx))
    return ptx.decode().rstrip("\0")


def count_ops(ptx):
    """{kernel name: Counter of KINDS} for every .entry in a PTX module."""
    kernels = {}
    for name, body in re.findall(r"\.entry\s+(\w+)\((.*?)\n}", ptx, flags=re.S):
        c = Counter()
        for line in body.splitlines():
            parts = line.split()[0].split(".") if line.strip() else [""]
            if parts[0] not in ("ld", "st") or len(parts) < 2 or parts[1] == "param":
                continue
            if parts[1] not in SPACES:
                c[parts[0]] += 1
            elif parts[1] == "global" and parts[2] == "nc":
                c["ld.global.nc"] += 1
            else:
                c[f"{parts[0]}.{parts[1]}"] += 1
        kernels[name] = c
    return kernels


def ops_table(title, kernels, names):
    lines = [title, f"  {'kernel':<16}" + "".join(f"{k:>13}" for k in KINDS)]
    for mangled, c in kernels.items():
        short = next(n for n in names if n in mangled)
        lines.append(f"  {short:<16}" + "".join(f"{c[k]:>13}" for k in KINDS))
    return lines


def cuda_cpp():
    src = (HERE / "cuda" / "memory_spaces.cu").read_text()
    arch = f"{CC[0]}{CC[1]}"
    _, log = nvrtc_compile(src, [f"-arch=sm_{arch}", "-std=c++17", "-Xptxas", "-v"])
    report = [l for l in log.splitlines() if "Compile time" not in l]
    write("ptxas-v.txt", "\n".join(report))
    prog, _ = nvrtc_compile(src, [f"-arch=compute_{arch}", "-std=c++17"])
    names = ["poly_constant", "poly_readonly", "bin_counts", "mirror_capped", "mirror"]
    return ops_table("CUDA C++ (NVRTC), cuda/memory_spaces.cu", count_ops(nvrtc_ptx(prog)), names)


def numba_kernels():
    nb = load("numba/memory_spaces.py", "memory_spaces")
    kernels = {}
    sigs = {"poly_constant": (float32[:], float32[:]), "bin_counts": (int32[:], int32[:, :])}
    for name, sig in sigs.items():
        ptx, _ = cuda.compile_ptx(getattr(nb, name).py_func, sig, cc=CC)
        kernels.update(count_ops(ptx))
    return ops_table("Numba (NVVM), numba/memory_spaces.py", kernels, list(sigs))


def triton_loads():
    tr = load("triton/cache_hints.py", "cache_hints")
    signature = {"x_ptr": "*fp32", "w_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr",
                 "TABLE": "constexpr", "X_CACHE": "constexpr", "X_EVICT": "constexpr", "W_EVICT": "constexpr"}
    aligned = [["tt.divisibility", 16]]
    attrs = {(0,): aligned, (1,): aligned, (2,): aligned}
    lines = [f"triton/cache_hints.py, BLOCK={tr.BLOCK}, TABLE={tr.TABLE}. Global loads in the PTX:"]
    for variant, (x_cache, x_evict, w_evict) in tr.VARIANTS.items():
        constexprs = {"BLOCK": tr.BLOCK, "TABLE": tr.TABLE, "X_CACHE": x_cache, "X_EVICT": x_evict, "W_EVICT": w_evict}
        src = triton.compiler.ASTSource(fn=tr.scale_by_table, signature=signature, constexprs=constexprs, attrs=attrs)
        ptx = triton.compile(src, target=GPUTarget("cuda", CC[0] * 10 + CC[1], 32)).asm["ptx"]
        hints = f"cache_modifier={x_cache!r}, x eviction_policy={x_evict!r}, w eviction_policy={w_evict!r}"
        lines += ["", f"{variant}: {hints}"]
        # Each distinct load instruction once; the operands differ only in registers.
        seen = []
        for line in ptx.splitlines():
            m = re.search(r"\b(ld\.global[\w.:]*)", line)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
        lines += [f"  {op}" for op in seen]
    return lines


def versions():
    v = metadata.version
    major, minor = nvrtc.nvrtcVersion()[1:]
    return [
        "Generated by generate.py. Do not edit by hand.",
        f"Target: sm_{CC[0]}{CC[1]} (compute capability {CC[0]}.{CC[1]})",
        f"Python {platform.python_version()}, NumPy {np.__version__}",
        f"NVRTC {major}.{minor} (nvidia-cuda-nvrtc {v('nvidia-cuda-nvrtc')}, cuda-bindings {v('cuda-bindings')})",
        f"numba-cuda {v('numba-cuda')}, triton {v('triton')}",
    ]


if __name__ == "__main__":
    os.chdir(HERE)
    OUT.mkdir(exist_ok=True)
    table = cuda_cpp() + [""] + numba_kernels()
    write("memory-ops.txt", "\n".join(table))
    write("triton-loads.txt", "\n".join(triton_loads()))
    write("VERSIONS.txt", "\n".join(versions()))
