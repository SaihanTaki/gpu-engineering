# Chapter 5: Occupancy

One kernel, `y = a * x + y` with a grid-stride loop, and the tools that report how many of its blocks fit on an SM.

- `occupancy_calc.py`: the occupancy rules from the chapter's simulation in Python. `blocks_per_sm(cc, regs, smem, block)` returns the resident blocks per SM and what each resource allows, for compute capability 7.5, 8.0 and 9.0. It follows `cuda_occupancy.h` from the CUDA Toolkit.
- `cuda/occupancy.cu`: `saxpy` and `saxpy_bounded`, the same kernel with `__launch_bounds__(256, 8)`. `main()` prints each kernel's registers from `cudaFuncGetAttributes`, asks `cudaOccupancyMaxActiveBlocksPerMultiprocessor` about several block sizes and shared memory sizes, takes the block size `cudaOccupancyMaxPotentialBlockSize` suggests, launches one full wave of blocks and checks the result. The Makefile passes `--resource-usage`, so the build also prints registers per kernel. It builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/occupancy.py`: the kernel compiled three ways: plain, with `max_registers=32` and with `launch_bounds=(256, 8)`. On a GPU it prints `get_regs_per_thread()` and `get_shared_mem_per_block()` for each and runs them through `occupancy_calc.py`.
- `triton/occupancy.py`: the kernel launched with `num_warps` from 1 to 16. On a GPU it prints the compiled kernel's `n_regs`, `n_spills` and `metadata.shared`, and the programs per SM those give.
- `test_correctness.py`: checks the calculator against hand-computed cases (the same ones as the simulation's Vitest tests), the Numba and Triton kernels against NumPy, and, with `nvcc` and a GPU, that the CUDA occupancy API agrees with the calculator.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/occupancy.py
python triton/occupancy.py
pytest
```

Register counts and occupancy queries need a real GPU. Under `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1` the scripts check the results and print a note instead. `pytest` sets both variables when `nvidia-smi` is missing and skips the CUDA test when `nvcc` is missing.
