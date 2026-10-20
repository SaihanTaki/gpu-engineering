# Chapter 11: Case study: matrix transpose

`out = in.T` for a `rows x cols` row-major float32 matrix, three ways. The CUDA and Numba kernels follow Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) (NVIDIA, 2013): 32 x 32 tiles, blocks of 32 x 8 threads, four elements per thread. Unlike the blog's version, every kernel here has bounds checks, so any shape works.

- `cuda/transpose_kernels.cuh`: `transpose_naive` and `transpose_shared<PAD>`. `PAD = 0` is the tiled kernel, `PAD = 1` the conflict-free one. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/transpose.cu`: times each kernel over 20 launches with `cudaEvent`, checks it against a CPU loop, and times a device-to-device `cudaMemcpy` of the same bytes as the ceiling. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/transpose.py`: the same three kernels. `make_shared(pad)` builds one kernel per padding, because a shared array's shape must be a compile-time constant.
- `triton/transpose.py`: one program per 32 x 32 tile, two ways: `tl.trans` on the loaded tile, and the same tile stored with swapped output strides. Neither mentions shared memory; the compiler adds it.
- `generate.py`: compiles both Triton kernels for `sm_80` without a GPU and writes their TTGIR, PTX and a count of shared memory instructions to `generated/`.
- `test_correctness.py`: checks every kernel against NumPy for shapes from 1 x 1 to 100 x 37, including ragged edge tiles in both directions. It also compiles the Numba kernels to real PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, which works on a machine with no GPU, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/transpose.py
python triton/transpose.py
pytest
```

Each program takes optional `rows cols`, for example `./cuda/transpose 4096 4096` (the default). The kernels index with `int`, so `rows * cols` must stay below 2<sup>31</sup>.

## Reading the output

Each line prints the mean time of one transpose and its effective bandwidth: `2 * rows * cols * 4` bytes (each element read once and written once) divided by that time. Compare the three kernels with each other and with the `memcpy` line, which moves the same number of bytes without transposing. The Python versions time with `time.perf_counter()` around 20 launches and a synchronize, so they also count launch overhead.

The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`. Pass a small size there, for example `python numba/transpose.py 64 96`.

To see where the time goes on a GPU, profile the CUDA program:

```sh
ncu --section MemoryWorkloadAnalysis_Tables ./cuda/transpose 4096 4096
```

The tables list the sectors per request for global loads and stores and the shared memory bank conflicts for each kernel.
