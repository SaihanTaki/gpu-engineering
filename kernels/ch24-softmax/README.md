# Chapter 24: Softmax and LayerNorm

Row softmax and LayerNorm forward over a row-major `[rows, cols]` float32 matrix, one row per block (or per warp, or per Triton program). The online softmax follows Maxim Milakov and Natalia Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018). LayerNorm uses Welford's update and the pairwise merge of Chan, Golub and LeVeque.

- `cuda/softmax_kernels.cuh`: `softmax_safe` (max pass, sum pass, normalize pass: three reads and one write per element), `softmax_online` (running max and sum in one pass, then normalize: two reads and one write), `softmax_online_warp` (the same with one warp per row, for short rows), and `layernorm_welford` (Welford statistics, then normalize with gamma and beta, in one kernel; it also writes each row's mean and 1/std). Block reductions use warp shuffles and one shared memory round, as in chapter 13. Device code only, so NVRTC can compile it without `nvcc`.
- `cuda/softmax.cu`: fills a matrix with normal values, puts 1000 into row 0 (its `exp` overflows float32) and gives row 1 a mean of 1000, runs every kernel, checks it against a double precision CPU reference and times it over 20 runs. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/softmax.py`: `softmax_safe`, `softmax_online` and `layernorm_welford` with 128 threads per block. They reduce through a shared memory tree instead of shuffles, so they also run on the Numba simulator.
- `triton/softmax.py`: the fused softmax of the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), one program per row with the whole row in one block, and a LayerNorm forward the same way. With the row on chip, each element is read once and written once.
- `test_correctness.py`: compares every kernel with NumPy in float64, for rows of 1 to 1000 columns. Stress rows prove stability: one value of 1000, a row of 1e4, a row near -200, and LayerNorm rows with mean 1000 and spread 1/3. Two tests show the failures the kernels avoid: naive `exp` overflowing to NaN, and `E[x^2] - E[x]^2` losing its digits in float32. It also compiles the Numba kernels and the CUDA header to PTX for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/softmax.py
python triton/softmax.py
pytest
```

Each program takes optional `rows cols`; the default is 4096 by 1024. It prints the mean time per launch and an effective bandwidth that counts the matrix read once and written once, `8 * rows * cols` bytes, the least any version can move.

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1`. The CUDA tests skip without `nvcc`. Times printed under the simulator or the interpreter mean nothing.
