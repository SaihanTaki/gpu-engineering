# Chapter 17: Sorting

Bitonic sort of float32 keys, ascending. The input is padded to a power of two (at least one 1024-key tile) with `+inf`, which sorts to the end and is dropped. NaN keys are not supported.

- `cuda/bitonic_kernels.cuh`: `bitonic_step`, one stage over the whole array with one thread per comparator, and `bitonic_shared`, every stage with `j < TILE` on a 1024-key tile in shared memory. Device code only, so NVRTC can compile it without `nvcc`.
- `cuda/bitonic.cu`: host loops `sort_global` (one launch per stage) and `sort_shared` (shared memory for `j < TILE`, global launches for the rest), checked against `std::sort` and timed next to `thrust::sort`. The Makefile builds for `sm_80`; set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/bitonic.py`: the same kernels and host loops. `make_shared(tile)` builds one shared memory kernel per tile size, because a shared array's size must be a compile-time constant.
- `triton/block_sort.py`: sorts every run of `BLOCK` keys with `tl.sort`. Triton has no barrier across programs, so this is the in-block part of a sort only.
- `test_correctness.py`: checks the Numba and Triton versions against `np.sort` for power-of-two and padded sizes, compiles the Numba kernels to PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/bitonic.py
python triton/block_sort.py
pytest
```

Each program takes an optional key count, for example `./cuda/bitonic 1000000` (the default is 2<sup>24</sup>). The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`; pass a small size there, for example `python numba/bitonic.py 200`.
