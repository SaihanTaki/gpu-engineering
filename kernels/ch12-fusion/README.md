# Chapter 12: Map and elementwise fusion

`out = relu(a * x + y)` on `n` float32 elements, two ways: three elementwise kernels (`scale`, `add`, `relu`) with two intermediate arrays in GPU memory, and one `fused` kernel that keeps the intermediates in registers. The three kernels move 7 arrays of DRAM traffic; the fused one moves 3.

- `cuda/fusion_kernels.cuh`: the four kernels. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/fusion.cu`: times one unfused pass (three launches) and one fused pass over 20 repetitions with `cudaEvent`, checks both against a CPU loop, and times back-to-back launches of an empty kernel, the fixed cost per launch on your machine. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/fusion.py`, `triton/fusion.py`: the same four kernels and the same timing.
- `generate.py`: compiles the CUDA header for `compute_80` with NVRTC, no GPU needed, and writes the PTX and a count of global loads, stores and FMAs per kernel to `generated/`.
- `test_correctness.py`: checks every version against NumPy for sizes from 1 to 3,000, including partial blocks. It compiles the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90` and the Numba kernels to real PTX, and checks that each kernel has the global loads and stores the chapter counts. It runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/fusion.py
python triton/fusion.py
pytest
```

Each program takes an optional `n`, for example `./cuda/fusion 4096`. The default is 2<sup>24</sup>.

## Reading the output

Each line prints the mean time of one pass, the bytes the chapter's count says it moves (7 or 3 arrays of `n * 4` bytes), and that count divided by the time. When the arrays fit in L2, the unfused pass can move fewer DRAM bytes than counted, so its effective bandwidth can come out above the GPU's peak. The Python versions time with `time.perf_counter()` around 20 passes and a synchronize, so they include Python's launch cost; the Triton unfused pass also allocates `t` and `u` each time, which PyTorch's caching allocator makes cheap but not free.

The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`. Pass a small size there, for example `python numba/fusion.py 4096`.

To see the DRAM traffic per kernel on a GPU:

```sh
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./cuda/fusion
```
