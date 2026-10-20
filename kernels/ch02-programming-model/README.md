# Chapter 2: Programming model

Every thread works out which element it owns from its block and thread index.

- `cuda/whoami.cu`: 1D kernel that records `blockIdx.x * blockDim.x + threadIdx.x`, plus a 2D kernel that maps `(col, row)` to a row-major index. `main()` checks both against a CPU loop.
- `numba/whoami.py`: the same two kernels with `cuda.grid(1)` and `cuda.grid(2)`.
- `triton/whoami.py`: the block-level equivalent. A Triton program owns a whole block of offsets, so there is only `tl.program_id`, no thread index.
- `test_correctness.py`: checks all three against NumPy, including sizes that are not a multiple of the block size.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/whoami.py
python triton/whoami.py
pytest
```

Without an NVIDIA GPU, set `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` to run the Python versions on the CPU. `pytest` sets both for you when `nvidia-smi` is missing, and skips the CUDA test when `nvcc` is missing.
