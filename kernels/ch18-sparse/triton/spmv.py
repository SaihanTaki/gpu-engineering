import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Nonzeros a program handles per loop iteration; tl.arange needs a power of two.
BLOCK = 32


@triton.jit
def spmv_csr_kernel(row_ptr, col, val, x, y, BLOCK: tl.constexpr):
    # One program per row: the CSR scalar idea, with a block of lanes inside the row.
    row = tl.program_id(0)
    start = tl.load(row_ptr + row)
    end = tl.load(row_ptr + row + 1)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # The trip count depends on the row, so programs of one launch run for different times.
    for i in range(start, end, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < end
        c = tl.load(col + offs, mask=mask, other=0)
        v = tl.load(val + offs, mask=mask, other=0.0)
        # A gather: c can be any column, so these addresses follow no pattern.
        acc += v * tl.load(x + c, mask=mask, other=0.0)
    tl.store(y + row, tl.sum(acc, axis=0))


def spmv(row_ptr, col, val, x):
    rows = row_ptr.numel() - 1
    y = torch.empty(rows, dtype=torch.float32, device=x.device)
    spmv_csr_kernel[(rows,)](row_ptr, col, val, x, y, BLOCK=BLOCK)
    return y


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    reps = 20
    g = torch.Generator().manual_seed(0)
    # Rows of 1 to 16 nonzeros in random columns.
    lengths = torch.randint(1, 17, (rows,), generator=g)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]).to(torch.int32)
    nnz = int(row_ptr[-1])
    col = torch.randint(0, rows, (nnz,), generator=g, dtype=torch.int32)
    val = torch.rand(nnz, generator=g)
    x = torch.rand(rows, generator=g)
    a = torch.sparse_csr_tensor(row_ptr.long(), col.long(), val, (rows, rows))
    want = (a @ x[:, None]).squeeze(1)

    row_ptr, col, val, x = (t.to(DEVICE) for t in (row_ptr, col, val, x))
    print(f"{rows} rows, {nnz} nonzeros, BLOCK = {BLOCK}")
    # The first call compiles the kernel, so it is not timed.
    y = spmv(row_ptr, col, val, x)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = spmv(row_ptr, col, val, x)
    sync()
    seconds = (time.perf_counter() - t0) / reps
    torch.testing.assert_close(y.cpu(), want, rtol=1e-4, atol=1e-4)
    print(f"  csr row per program {seconds * 1e3:9.3f} ms")
    print("OK")
