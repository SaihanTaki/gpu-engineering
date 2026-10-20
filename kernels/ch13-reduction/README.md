# Chapter 13: Reduction

The sum of a float32 array, several ways. The shared memory kernels follow the first steps of Mark Harris, [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (NVIDIA): interleaved addressing, then sequential addressing, then a last warp that finishes without barriers. Here that last warp uses `__shfl_down_sync`, which did not exist when the slides were written.

- `cuda/reduce_kernels.cuh`: `reduce_interleaved`, `reduce_sequential` and `reduce_warp_final` each reduce one block's slice in shared memory and write a partial sum per block. `reduce_atomic` uses a grid-stride loop, a shuffle reduction in every warp and one `atomicAdd` per block, so one launch produces the total. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/reduce.cu`: runs the three multi-pass kernels again on their own partial sums until one value is left, and the atomic kernel once with 4 blocks per SM. It times each method over 20 runs with `cudaEvent` and checks it against a CPU sum. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/reduce.py`: the same four kernels with 256 threads per block.
- `triton/reduce.py`: `tl.sum` over a block of 1024 elements, either written as one partial sum per program and reduced again, or added into one value with `tl.atomic_add`.
- `generate.py`: compiles the CUDA header with NVRTC and the Triton kernels with Triton's own `ptxas`, both for `sm_80` and without a GPU, and writes a count of shuffles, barriers, shared memory accesses and atomics per kernel to `generated/summary.txt`, plus the Triton PTX.
- `test_correctness.py`: checks every method against NumPy for n from 1 to 70,000. The inputs are whole numbers from -3 to 3, so a float32 sum is exact in any order and the test can ask for equality. One extra test uses random normal floats with a tolerance. It also compiles the Numba kernels and the CUDA header to PTX for `sm_75`, `sm_80` and `sm_90`, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/reduce.py
python triton/reduce.py
pytest
```

Each program takes an optional element count; the default is 2<sup>22</sup>. It prints the mean time of one complete sum and an effective bandwidth, `4 * n` bytes over that time.

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1`. The Numba simulator has no `shfl_down_sync`, so `reduce_warp_final` and `reduce_atomic` in Numba only run on a GPU; on a CPU they are compiled to PTX instead. Times printed under the simulator or the interpreter mean nothing.

## Float sums depend on order

Float addition is not associative, so the tree's order, NumPy's order and an atomic kernel's order give slightly different sums for real-valued data. The atomic order changes from run to run, so `reduce_atomic` and `sum_atomic` can return different last bits on two runs over the same input. Whole-number inputs below 2<sup>24</sup> avoid that, which is why the tests and the programs use them.
