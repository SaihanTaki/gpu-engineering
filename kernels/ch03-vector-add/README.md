# Chapter 3: First kernel: vector add

`c = a + b`, with every step of the trip spelled out: host arrays, device arrays, copy in, launch, copy out, check.

- `cuda/vector_add.cu`: the kernel `vector_add(a, b, c, n)` with a bounds check. `main()` fills the host arrays, allocates `d_a`, `d_b`, `d_c` with `cudaMalloc`, copies `a` and `b` in with `cudaMemcpy`, launches 256 threads per block on a rounded-up grid, checks `cudaGetLastError()` and `cudaDeviceSynchronize()`, copies `c` back, checks it against a CPU loop and frees the device arrays. It exits with 1 on a mismatch. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/vector_add.py`: the same kernel with `cuda.grid(1)`. `run(a, b)` does the copies with `cuda.to_device` and `copy_to_host`.
- `triton/vector_add.py`: the block-level kernel with `BLOCK = 1024` and a mask. `vector_add(a, b)` takes torch tensors that are already on the device.
- `test_correctness.py`: checks all three against NumPy for n = 1, 1000, 1025 and 2<sup>20</sup>, so the last block is ragged for both 256 and 1024.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/vector_add.py
python triton/vector_add.py
pytest
```

Each program takes an optional element count, for example `./cuda/vector_add 1000000`. The default is 2<sup>24</sup>.

## Reading the output

Each program prints three phases:

- `copy in`: `a` and `b` from host memory to device memory, over PCIe or NVLink.
- `kernel`: the launch until it finishes on the GPU.
- `copy out`: `c` from device memory back to the host.

Each line also shows the effective bandwidth: bytes moved in that phase divided by its time. The copies move 2n and n floats. The kernel reads `a` and `b` and writes `c`, so it counts 3n floats. Compare the copy numbers with your link's bandwidth and the kernel number with your GPU's memory bandwidth. Comparing the sum of the two copy times with the kernel time shows how much of the run is moving data rather than computing.

The CUDA version times with `cudaEvent`. The Python versions use `time.perf_counter()` around each phase with a synchronize at the end, which also counts Python overhead. All three skip the first launch, which pays for compiling or loading the kernel.

Every number here is only meaningful on a real GPU. Under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1` the kernel runs on the CPU in Python and the "copies" stay in host memory, so the times and GB/s figures mean nothing. Pass a small count there. `pytest` sets both variables for you when `nvidia-smi` is missing, and skips the CUDA tests when `nvcc` is missing.
