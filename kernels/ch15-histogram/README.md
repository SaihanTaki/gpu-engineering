# Chapter 15: Histogram and atomics

A 256-bin histogram of `n` bytes, two ways: one global atomic per element, and a private copy per block in shared memory that is merged into the global histogram at the end. Every version is checked exactly against `np.bincount`.

- `cuda/histogram_kernels.cuh`: `hist_global` and `hist_shared`. Both use a grid-stride loop, so the grid size does not depend on `n`. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/histogram.cu`: builds three inputs (uniform bytes, a skewed one where byte `b` has probability 2<sup>-(b+1)</sup>, and every byte equal), times each kernel over 20 launches with `cudaEvent` and checks it against a CPU count. The grid is 8 blocks per SM of 256 threads. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/histogram.py`: the same two kernels with `cuda.atomic.add`.
- `triton/histogram.py`: one program per 1,024 bytes. `atomic` calls `tl.atomic_add` once per element; `local` builds the program's histogram with `tl.histogram` and adds it with one `tl.atomic_add` per bin. Both use `sem="relaxed"`, the ordering of CUDA's `atomicAdd`; Triton's default is `acq_rel`.
- `generate.py`: compiles both Triton kernels for `sm_80` without a GPU and writes their PTX and a count of atomic instructions to `generated/`.
- `test_correctness.py`: checks every kernel against `np.bincount` for the three inputs and sizes from 1 to 3,073 bytes. It also compiles the Numba kernels to real PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, which works on a machine with no GPU, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/histogram.py
python triton/histogram.py
pytest
```

Each program takes an optional `n`, for example `./cuda/histogram 16777216` (the default, 2<sup>24</sup>).

## Reading the output

Each line prints the mean time of one histogram and its throughput, `n` bytes read over that time. Compare the two kernels on each input, and each kernel across the inputs. The Python versions time with `time.perf_counter()` and a synchronize, so they also count launch overhead; the Triton one also counts allocating and zeroing the output.

The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`. Pass a small size there, for example `python numba/histogram.py 10000`.

To see the atomics on a GPU, profile the CUDA program:

```sh
ncu --section MemoryWorkloadAnalysis_Tables ./cuda/histogram
```
