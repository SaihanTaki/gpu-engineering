# Chapter 14: Scan (prefix sum)

Inclusive and exclusive prefix sums of int32 arrays. Every version scans one tile per block (or Triton program), scans the tile totals with the same code, recursively, and adds each tile's offset back: scan-then-add.

- `cuda/scan_kernels.cuh`: `scan_hillis_steele<INCLUSIVE>` (256 threads, one element each), `scan_blelloch<INCLUSIVE>` (256 threads, 512 elements, after Harris, Sengupta and Owens, GPU Gems 3, chapter 39, without the bank conflict padding) and `add_offsets`. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/scan.cu`: the recursive driver. Times all four variants over 20 runs with `cudaEvent` and checks each against a CPU loop. Scratch space for the tile sums is allocated once, outside the timing. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/scan.py`: the same kernels and driver. The Python driver allocates the tile sums on every call, so its times include those allocations.
- `triton/scan.py`: one program per 1,024 elements, scanned with `tl.associative_scan` or `tl.cumsum`; the exclusive result is `inclusive - x`.
- `test_correctness.py`: checks every version against `np.cumsum` for sizes from 1 to 5,000, including exact multiples of the tile sizes and one past them. Triton is also run with 16 elements per program, so the recursion goes three levels deep. On a GPU, Numba also runs 70,000 and 300,000 elements (two levels). The test compiles the Numba kernels to PTX and the CUDA header with NVRTC for `sm_75`, `sm_80` and `sm_90`, which works without a GPU, and runs the CUDA program when `nvcc` is present.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/scan.py
python triton/scan.py
pytest
```

Each program takes an optional size, for example `./cuda/scan 1000000`; the default is 2<sup>24</sup>. Inputs are 0 to 9, so `n` must stay below 2<sup>31</sup> / 9 for the sums to fit in an int.

## Reading the output

Each line prints the mean time of one complete scan and its effective bandwidth: `2 * n * 4` bytes (each element read once and written once, the minimum) divided by that time. A scan-then-add moves about twice that, so on a memory-bound GPU expect it to reach about half of the memory bandwidth at best. That is an estimate from the traffic count, not a measurement.

The times mean nothing under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1`. Pass a small size there, for example `python numba/scan.py 5000`.

To see the launches and the bank conflicts of the Blelloch kernel:

```sh
nsys profile --stats=true ./cuda/scan
ncu --kernel-name regex:scan_blelloch --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./cuda/scan
```
