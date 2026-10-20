# Chapter 18: Sparse matrices

`y = A x` for a sparse float32 matrix `A`, in the CSR and ELL formats. The CSR kernels follow Bell and Garland, [Efficient Sparse Matrix-Vector Multiplication on CUDA](https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf) (NVIDIA Technical Report NVR-2008-004, 2008), with `__shfl_down_sync` in place of their shared memory reduction.

- `cuda/spmv_kernels.cuh`: `spmv_csr_scalar` (one thread per row), `spmv_csr_vector` (one warp per row, a shuffle reduction at the end) and `spmv_ell` (one thread per row, column-major ELL). The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/spmv.cu`: builds two matrices with as many rows as columns, `uniform` (1 to 16 nonzeros per row) and `skewed` (the same, plus every 1024th row with `cols / 8` nonzeros), times each kernel over 20 launches with `cudaEvent` and checks it against a CPU loop. ELL is skipped when padding would need more than 4 slots per nonzero, which happens for `skewed`. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/spmv.py`: the same three kernels, plus `csr_from_dense` and `ell_from_csr` in NumPy.
- `triton/spmv.py`: one program per CSR row, a masked loop over the row in blocks of 32 nonzeros.
- `test_correctness.py`: checks every kernel against `scipy.sparse` if it is installed, else a dense NumPy product, on banded, random, power-law (row 0 longer than a warp) and empty-row matrices. It compiles the Numba vector kernel to real PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, which works without a GPU, and runs the CUDA program when `nvcc` is present.

The Numba simulator (`NUMBA_ENABLE_CUDASIM=1`) has no warp shuffles. The test adds one built on block barriers, which works because every warp of `spmv_csr_vector` either leaves at once or runs all five shuffles. Run as a script under the simulator, `numba/spmv.py` skips the vector kernel.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/spmv.py
python triton/spmv.py
pytest
```

Each program takes an optional row count, for example `./cuda/spmv 1048576` (the default). Under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`, pass a small one such as 64; the times mean nothing there.

## Reading the output

Each line prints the mean time of one SpMV. Compare the three kernels on `uniform`, where rows are short and similar, with `skewed`, where a few rows are long. The ELL line prints its width and the number of slots per nonzero, the cost of padding.

To see where the time goes, profile the CUDA program:

```sh
ncu --section MemoryWorkloadAnalysis_Tables --section WarpStateStats ./cuda/spmv
```

Look at the sectors per request of the global loads, which are high for `spmv_csr_scalar` and low for `spmv_ell` and `spmv_csr_vector`.
