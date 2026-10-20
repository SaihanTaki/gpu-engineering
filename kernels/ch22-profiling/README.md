# Chapter 22: Profiling and debugging

Five bugs, each next to its fix, plus the tools that find them. Every fixed version is checked against a CPU or NumPy result; the buggy versions are there to be run under `compute-sanitizer`, the Numba simulator and the Triton interpreter.

- `cuda/bugs_kernels.cuh`: the kernels, device code only, so NVRTC can compile them without `nvcc`.
  - `count_racy` / `count_atomic`: count positive elements with `count += 1` on a shared counter, or with `atomicAdd`. Found by racecheck.
  - `diff_oob` / `diff`: a forward difference whose last thread reads `in[n]`. Found by memcheck.
  - `reverse_nosync` / `reverse`: reverse each block through shared memory, without and with `__syncthreads()`. Found by racecheck.
  - `add_index`: correct, but the host clears it with `cudaMemset(d_v, 0, n)`, `n` bytes instead of `n` ints. Found by initcheck.
  - `block_sum_divergent` / `block_sum`: a `__syncthreads()` in each branch of an `if`, against one barrier that every thread reaches. Found by synccheck.
- `cuda/bugs.cu`: runs `./bugs race|oob|nosync|uninit|divergent|launch|all [n]`. `launch` starts a kernel with 2,048 threads per block, over the limit of 1,024, and shows that nothing reports it unless the program calls `cudaGetLastError()`. The `CUDA_CHECK` and `CUDA_CHECK_LAUNCH` macros check every runtime call and every launch. Each bug runs inside an NVTX range named after it. The program exits with 1 only if a fixed version is wrong.
- `cuda/Makefile`: builds with `-lineinfo` so the tools can name source lines. Targets `memcheck`, `racecheck`, `initcheck`, `synccheck` and `sanitize` (all four) run each tool on the bug it finds; `nsys` and `ncu` profile the program.
- `numba/bugs.py`: the racy and atomic counters, a kernel with no bounds check, and one that reads `x[-1]`.
- `triton/bugs.py`: a scale kernel with and without `mask=`. Both buffers have a guard band of known values after them, so an access past the end changes memory the test can check.
- `generate.py`: compiles the CUDA header with NVRTC for `sm_80` and writes the PTX of both counters to `generated/`.
- `test_correctness.py`: checks every fixed version, checks that the bugs show up where they should (the unmasked Triton store hits the guard band; the Numba simulator raises `IndexError`), compiles the Numba counters to PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
make -C cuda sanitize
python numba/bugs.py
python triton/bugs.py
pytest
```

Without a GPU:

```sh
NUMBA_ENABLE_CUDASIM=1 python numba/bugs.py
TRITON_INTERPRET=1 python triton/bugs.py
```

## What to expect

- `./bugs all` prints `OK` or `WRONG` for each buggy and fixed version. A buggy version can print `OK`: fresh device memory often holds zeros, a read one past the end usually lands in memory that exists, and a missing barrier only matters when the warps happen to run in the wrong order. That is why the tools exist.
- `make memcheck` uses `--padding 32`, because, as the Compute Sanitizer guide explains, two buffers can sit back to back, and a read past the end of one then lands in the other without an error.
- racecheck checks shared memory only. A race on a counter in global memory is invisible to it.
- The Numba simulator runs each GPU thread as a Python thread, and Python rarely switches threads in the middle of `count[0] += 1`, so `count_racy` almost always gives the right count there. On a GPU it can lose updates.
- The Triton interpreter does not check bounds either: the unmasked kernel writes into the guard band without an error. It runs the kernel as Python, so `print`, `tl.device_print` and `pdb` work inside it. Pass `show=True` to `scale()` to print the offsets or the mask of each program.

Only the device code has been compiled so far, with NVRTC. `test_cuda_binary` builds and runs the whole program wherever `nvcc` is present.
