# Chapter 25: FlashAttention

`O = softmax(Q Kᵀ / sqrt(d)) V` for every (batch, head), computed tile by tile so the `n x n` score matrix never reaches GPU memory. Both kernels follow the FlashAttention-2 loop order: one program (Triton) or thread block (CUDA) per block of Q rows, looping over blocks of K and V with an online softmax.

- `triton/flash_attention.py`: the main kernel. `flash_attention(q, k, v, causal)` takes `(batch, heads, n, d)` tensors with `d` a power of two of at least 16, and launches one program per (Q block, batch * head). Both products use `tl.dot`, so FP16 inputs run on tensor cores; the accumulator, the row max `m` and the row sum `l` stay in FP32. With `causal=True` the K/V loop stops at the block's last row, skipping fully masked tiles. `naive_attention` is the reference that materializes `S` and `P`, as Algorithm 0 of Dao et al. 2022 does. Run as a script, it times both and `torch.nn.functional.scaled_dot_product_attention`.
- `cuda/flash_attention_kernels.cuh`: a simple FP32 version without tensor cores. One thread owns one row of Q (`d = 64`, fixed so its arrays stay in registers); K and V tiles of 32 rows go through shared memory. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/flash_attention.cu`: runs 8 heads, times the kernel with `cudaEvent` over 10 launches, checks two heads against a CPU reference in double, and exits with 1 on a mismatch. Arguments `[n] [causal]`, default `1024 0`. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `test_correctness.py`: checks the Triton kernel against NumPy for `n` from 1 to 64, `d` of 16 and 32, causal and not, in FP32 and FP16, with equal and unequal block sizes. It compiles the CUDA header with NVRTC for `compute_75`, `compute_80` and `compute_90` and checks the PTX: shared-memory loads and stores, no `mma` instructions, and global stores only for the 64 output values of a row. It runs the CUDA program when `nvcc` is present.

There is no Numba version.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python triton/flash_attention.py
pytest
```

## Reading the output

The Triton script prints the mean time of each version and the rate in TFLOPS from the FlashAttention-2 paper's count, `4 n² d` FLOPs per head for the forward pass, halved with the causal mask. `naive` and `torch sdpa` include PyTorch's dispatch cost. Without a GPU, `TRITON_INTERPRET=1` runs the kernel in NumPy on the CPU: the results are checked, the times mean nothing, and a small size such as `python triton/flash_attention.py 128` finishes in reasonable time.

The CUDA program is a teaching version: one thread per row, FP32 FMAs, no tensor cores. Expect it to be far slower than the Triton kernel in FP16.
