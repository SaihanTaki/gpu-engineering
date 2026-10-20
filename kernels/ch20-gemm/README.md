# Chapter 20: Matrix multiply, step by step

`C = A @ B` for row-major float32 matrices, `A` is `M x K`, `B` is `K x N`. The CUDA kernels follow the steps of Simon Boehm's worklog, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) (2022), with bounds checks added so any shape works.

- `cuda/gemm_kernels.cuh`: six kernels, device code only, so NVRTC can compile it without `nvcc`.
  1. `gemm_naive`: one thread per element of C, `threadIdx.x` walks rows (uncoalesced A).
  2. `gemm_coalesced`: the same with `threadIdx.x` walking columns.
  3. `gemm_smem<BM, BN, BK>`: shared memory tiles of A and B, one output per thread.
  4. `gemm_1d<BM, BN, BK, TM>`: each thread computes `TM` outputs of one column.
  5. `gemm_2d<BM, BN, BK, TM, TN>`: each thread computes a `TM x TN` block from register copies of A and B.
  6. `gemm_vec<BM, BN, BK, TM, TN>`: `gemm_2d` with `float4` global and shared loads and A's tile stored transposed. Needs `K` and `N` to be multiples of 4.
- `cuda/gemm.cu`: times every kernel over 10 launches with `cudaEvent`, autotunes `gemm_vec` over six tile shapes by timing each, times `cublasSgemm` as the reference, and checks every result exactly against a CPU loop (every element for small sizes, 4096 sampled elements for large ones). The inputs are whole numbers from -4 to 4, so float32 sums are exact in any order. The Makefile builds for `sm_80` (A100) and links cuBLAS; set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/gemm.py`: naive (coalesced), shared memory tiled (16 x 16), and register-blocked (`make_regblock(BM, BN, BK, TM, TN)`, default 64 x 64 x 8 with 4 x 4 per thread). Shared and local array shapes must be compile-time constants, so each tile shape is its own kernel.
- `triton/gemm.py`: one `tl.dot` kernel with `@triton.autotune` over six block shapes, keyed on `(M, N, K)`. `input_precision="ieee"` keeps it on full float32; the default on Ampere and later is TF32 on tensor cores.
- `generate.py`: compiles the CUDA kernels with NVRTC and one Triton config for `sm_80`, without a GPU, and writes PTX instruction counts to `generated/summary.txt`.
- `test_correctness.py`: checks every Numba kernel and the Triton kernel against NumPy for shapes from 1 x 1 x 1 to 70 x 65 x 40, compiles the Numba kernels to real PTX, compiles the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90` (and checks that `gemm_vec` uses only 16-byte loads), and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/gemm.py
python triton/gemm.py
pytest
```

Each program takes optional `M N K`, for example `./cuda/gemm 4096 4096 4096` (the default for CUDA and Triton; Numba defaults to 2048). Without a GPU, the tests switch to `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` on their own. The interpreter cannot time configs, so there the Triton autotuner gets a single config. Times printed in those modes mean nothing.

## Reading the output

Each line prints the mean time of one multiply and `2 * M * N * K` floating point operations divided by that time, in GFLOP/s. Compare each step with the previous one and with the cuBLAS line. The Python versions time with `time.perf_counter()` around 10 calls and a synchronize, so they also count launch overhead; the Triton script compares against `torch.matmul` with TF32 turned off.

To see what limits a kernel, profile it:

```sh
ncu --set full -k regex:gemm_ ./cuda/gemm 4096 4096 4096
```

Look at the memory throughput against compute throughput (the roofline section), the shared memory bank conflicts, and the warp stall reasons.
