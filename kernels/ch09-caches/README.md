# Chapter 9: Registers, caches and other memory

Kernels that put data in each memory space, and the compiler reports that show where it went.

- `cuda/memory_spaces.cu`: `poly_constant` (polynomial coefficients in `__constant__` memory), `poly_readonly` (the same with `const __restrict__` pointers and `__ldg`), `bin_counts` (a per-thread array indexed by data, which lands in local memory), and `mirror` / `mirror_capped` (one kernel that needs 96 registers, and a copy limited to 32 with `__maxnreg__(32)` that spills). `main()` checks every kernel against a CPU loop and prints registers and local memory from `cudaFuncGetAttributes`. The Makefile passes `-Xptxas -v`, so the build prints the spill report. `__maxnreg__` needs CUDA 12.4 or later.
- `numba/memory_spaces.py`: `poly_constant` with `cuda.const.array_like` and `bin_counts` with `cuda.local.array`. On a GPU it prints `get_regs_per_thread()` and `get_local_mem_per_thread()`.
- `triton/cache_hints.py`: one kernel compiled with different `cache_modifier` and `eviction_policy` arguments to `tl.load`.
- `generate.py`: compiles all three for sm_80 without a GPU or `nvcc` and writes `generated/`: the `-Xptxas -v` report of the CUDA file (from NVRTC), a count of PTX memory instructions per kernel, and the load instructions Triton emits for each set of hints. Versions are in `generated/VERSIONS.txt`.
- `test_correctness.py`: checks the Numba and Triton kernels against NumPy and PyTorch, checks the committed ptxas report still shows the spill, and runs the CUDA binary when `nvcc` is present.

Inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/memory_spaces.py
python triton/cache_hints.py
python generate.py
pytest
```
