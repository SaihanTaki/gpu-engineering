# Chapter 16: Stencil and convolution

2D convolution of an `h x w` float32 image with a `(2R+1) x (2R+1)` filter:

```
out[y][x] = sum over dy, dx in [-R, R] of f[dy + R][dx + R] * in[y + dy][x + dx]
```

**Boundary rule:** pixels outside the image count as zero (zero padding), so `out` has the same size as `in`. The filter is not flipped, as in deep learning libraries; for a symmetric filter that is the same as a textbook convolution. `test_correctness.py` holds the NumPy reference and a test that pins this rule down.

- `cuda/conv2d_kernels.cuh`: `conv2d_naive<R>` (one thread per output, whole window from global memory) and `conv2d_tiled<R>` (a 16 x 16 block loads its 16 x 16 output tile plus an `R`-pixel halo into shared memory, then reads the window from there). Both read the filter from `__constant__ float c_filter[]`. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/conv2d.cu`: fills the filter with `cudaMemcpyToSymbol`, times both kernels for R = 1, 2, 3 over 20 launches with `cudaEvent`, and checks them against a CPU loop. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/conv2d.py`: the same two kernels. `make_kernels(filt)` builds them for one filter, because `cuda.const.array_like` and the shared array shape are fixed when the kernel compiles.
- `triton/conv2d.py`: one program per 16 x 32 output tile, `(2R+1)^2` masked loads of shifted tiles. `other=0.0` on the mask is the zero padding.
- `generate.py`: compiles all kernels for `sm_80` with R = 2, without a GPU, and writes PTX instruction counts to `generated/summary.txt`.
- `test_correctness.py`: checks every kernel against NumPy for R = 1, 2, 3 and shapes from 1 x 1 to 40 x 21, compiles the Numba kernels to real PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/conv2d.py
python triton/conv2d.py
pytest
```

Each program takes optional `h w`, for example `./cuda/conv2d 4096 4096` (the default is 2048 x 2048). The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`; pass a small size there, for example `python numba/conv2d.py 64 64`.

Each line prints the mean time of one convolution and an effective bandwidth that counts each pixel read once and written once.
