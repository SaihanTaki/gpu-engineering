"""Compile matmul_kernel to PTX for sm_75, sm_80 and sm_90 and count its matrix instructions.

Needs no GPU: Triton compiles for any target it knows. Run without TRITON_INTERPRET.
"""
import collections
import importlib.util
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

spec = importlib.util.spec_from_file_location("ch23_matmul", Path(__file__).with_name("matmul.py"))
matmul = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matmul)

BLOCK = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}
INTS = ["M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn"]


def ptx(dtype, precision, cc):
    sig = {"a_ptr": f"*{dtype}", "b_ptr": f"*{dtype}", "c_ptr": "*fp32", **{n: "i32" for n in INTS}}
    for name, value in BLOCK.items():
        sig[name] = "constexpr"
    sig["INPUT_PRECISION"] = "constexpr"
    src = ASTSource(fn=matmul.matmul_kernel, signature=sig, constexprs={**BLOCK, "INPUT_PRECISION": precision})
    return triton.compile(src, target=GPUTarget("cuda", cc, 32)).asm["ptx"]


def matrix_ops(text):
    # The opcode of every multiply-accumulate instruction: tensor core (mma, wgmma) or FP32 FMA.
    ops = (line.split()[0].rstrip(";") for line in text.splitlines() if line.strip() and not line.lstrip().startswith("//"))
    return collections.Counter(op for op in ops if op.startswith(("mma.", "wgmma.mma", "fma.rn.f32")))


if __name__ == "__main__":
    print(f"Triton {triton.__version__}, tile {BLOCK['BLOCK_M']} x {BLOCK['BLOCK_N']} x {BLOCK['BLOCK_K']}, 4 warps")
    for dtype, precision in [("fp16", "ieee"), ("fp32", "tf32"), ("fp32", "ieee")]:
        for cc in (75, 80, 90):
            counts = matrix_ops(ptx(dtype, precision, cc))
            body = ", ".join(f"{op} x{n}" for op, n in counts.most_common(2)) or "none"
            print(f"{dtype} {precision:4s} sm_{cc}: {body}")
