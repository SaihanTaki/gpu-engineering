# Chapter 23: Tensor cores and mixed precision

`C = A B` with FP16 inputs and an FP32 accumulator, three ways.

- `cuda/wmma_gemm.cuh`: two WMMA kernels. Each warp computes one 16 x 16 tile of `C` with `load_matrix_sync`, `mma_sync` and `store_matrix_sync` on 16 x 16 x 16 fragments; a block of 4 warps covers 32 x 32. `wmma_gemm_f32acc` accumulates in FP32, `wmma_gemm_f16acc` in FP16. The matrices are row-major and every size must be a multiple of 16. The tiles are loaded straight from global memory with no shared-memory staging, so this is the API, not a fast GEMM.
- `cuda/wmma_gemm.cu`: builds random FP16 matrices, times both kernels over 20 launches with `cudaEvent`, prints TFLOPS, and checks them against an FP64 CPU product of the same FP16 inputs. The error is measured against the sum of `|a| |b|` over k. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100. WMMA needs `sm_70` or later.
- `numba/fp16_gemm.py`: numba-cuda 0.30.4 has no WMMA or `mma` API, so these kernels run on the CUDA cores, one thread per element of `C`: `gemm_f16_f32acc` converts FP16 inputs to FP32 and sums in FP32, `gemm_f16_f16acc` sums in FP16 with `cuda.fp16.hfma`. The script also adds 4,096 ones in each, which shows an FP16 sum stopping at 2,048.
- `triton/matmul.py`: one `tl.dot` loop with an FP32 accumulator. FP16 and BF16 inputs go to the tensor cores; for FP32 inputs, `input_precision="tf32"` (the default on NVIDIA GPUs) rounds them to TF32 for the tensor cores and `"ieee"` keeps full FP32 on the CUDA cores. `allow_tf32` is the deprecated name for the same switch.
- `triton/show_ptx.py`: compiles `matmul_kernel` for `sm_75`, `sm_80` and `sm_90` without a GPU and counts its matrix instructions. `triton/show_ptx.out.txt` is its output with Triton 3.8.0; the test checks that the two still agree.
- `test_correctness.py`: checks every kernel against an FP64 NumPy product with error bounds from the accumulator's precision, on sizes that do not fill a block. It compiles the Numba kernels to real PTX, the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

NVRTC compiles `wmma_gemm.cuh` only with an include path that has `mma.h` and `crt/mma.h`. The pip package `nvidia-cuda-runtime` ships `mma.h` without `crt/`, so the test uses `$CUDA_HOME/include` or the headers bundled with Triton.

Without a GPU: Numba runs on its simulator (`NUMBA_ENABLE_CUDASIM=1`), whose `hfma` rounds the product and the sum separately, and Triton runs in its interpreter (`TRITON_INTERPRET=1`), which ignores `input_precision` and computes BF16 `tl.dot` wrongly, so the BF16 test is skipped there.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/fp16_gemm.py
python triton/matmul.py
python triton/show_ptx.py
pytest
```

Each program takes an optional size, for example `./cuda/wmma_gemm 2048` (default 1024; the CPU reference grows as n^3). Under the simulator or interpreter, pass a small one such as 32; the times mean nothing there.

## Reading the output

Compare the FP32 and FP16 accumulate lines of the CUDA program: on the T4, A100 and H100 NVIDIA lists the same tensor rate for both, so the FP16 accumulator buys no speed there and costs accuracy. To see whether the tensor cores ran, profile:

```sh
ncu --section ComputeWorkloadAnalysis --section InstructionStats ./cuda/wmma_gemm
```

Look for HMMA instructions in the instruction statistics and for the tensor pipe in the compute workload section.
