import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tl.arange needs a power of two.
BLOCK = 1024


@triton.jit
def partial_sums_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Masked-off elements read as 0, which leaves a sum unchanged.
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + tl.program_id(0), tl.sum(x, axis=0))


@triton.jit
def atomic_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Grid-stride loop: each program sums many blocks before it reduces once.
    for start in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(acc, axis=0))


def sum_two_pass(x):
    # Each launch leaves one partial sum per program; launch again on those.
    while True:
        programs = triton.cdiv(x.numel(), BLOCK)
        out = torch.empty(programs, dtype=torch.float32, device=x.device)
        partial_sums_kernel[(programs,)](x, out, x.numel(), BLOCK=BLOCK)
        if programs == 1:
            return out[0]
        x = out


def sum_atomic(x, programs=None):
    if programs is None:
        programs = 4 * torch.cuda.get_device_properties(x.device).multi_processor_count if x.is_cuda else 4
    out = torch.zeros(1, dtype=torch.float32, device=x.device)
    atomic_sum_kernel[(min(programs, triton.cdiv(x.numel(), BLOCK)),)](x, out, x.numel(), BLOCK=BLOCK)
    return out[0]


METHODS = {"two_pass": sum_two_pass, "atomic": sum_atomic}


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 22
    # Whole numbers from -3 to 3: the float32 sum is exact, so every order agrees.
    x = torch.randint(-3, 4, (n,), generator=torch.Generator().manual_seed(0)).to(torch.float32)
    expected = int(x.to(torch.int64).sum())
    x = x.to(DEVICE)
    reps = 20
    print(f"n = {n} floats, BLOCK = {BLOCK}, expected sum {expected}")
    for name, fn in METHODS.items():
        # The first call compiles the kernel, so it is not timed.
        fn(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            got = fn(x)
        sync()
        seconds = (time.perf_counter() - t0) / reps
        assert float(got) == expected, (name, float(got), expected)
        print(f"  {name:<9} {seconds * 1e3:9.3f} ms  {4 * n / seconds / 1e9:8.1f} GB/s")
    print("OK")
