# Chapter 7: Shared memory

One matrix multiply, `C = A * B` for square `n x n` float32 matrices, written twice: once reading every operand from global memory, and once staging tiles of A and B in shared memory.

- `cuda/matmul.cu`: `matmul_naive` and `matmul_tiled` with 16 x 16 thread blocks. The tiled kernel declares two `__shared__` tiles and calls `__syncthreads()` after loading them and again after using them. Threads past the edge of the matrix load zeros, so any `n` works. `main()` runs both kernels, times each with CUDA events, checks them against a CPU loop and exits with 1 on a mismatch. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/matmul.py`: the same two kernels with `cuda.shared.array` and `cuda.syncthreads()`.
- `triton/matmul.py`: one kernel that loads a 32 x 32 tile of A and of B per loop iteration and multiplies them with `tl.dot`. It has no shared memory or barriers in the source; the compiler adds them.
- `generate.py`: compiles the tiled Numba kernel and the Triton kernel for sm_80 without a GPU and writes `generated/shared_ops.txt`, the shared memory declarations, loads, stores and barriers each compiler produced. The full Numba PTX and Triton TTGIR are next to it.
- `test_correctness.py`: checks every version against NumPy for n = 1, 7, 32, 50 and 256, and checks that the generated files exist.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/matmul.py
python triton/matmul.py
python generate.py
pytest
```

`./cuda/matmul` takes an optional `n`; the default is 1024. The CPU reference is a plain triple loop, so very large `n` takes a while to check.

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1`, leaves out n = 256 for Numba (the simulator runs one Python thread per GPU thread) and skips the CUDA test when `nvcc` is missing.
