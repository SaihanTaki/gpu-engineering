# Chapter 8: Bank conflicts

Row sums of a matrix whose column count is a multiple of 32. Each block is one warp and handles 32 rows. The warp loads a 32 x 32 tile into shared memory with coalesced reads, then every lane sums its own row of the tile. At each step of that sum the warp reads one column of the tile, which is the access that causes bank conflicts.

- `cuda/row_sums.cu`: `row_sums<0>` (plain `float tile[32][32]`, a 32-way conflict on every column read), `row_sums<1>` (`tile[32][33]`, conflict-free) and `row_sums_swizzled` (`tile[32][32]` with element `(r, c)` stored in column `c ^ r`, conflict-free). `main()` checks all three against a CPU loop and prints the time of each. It takes optional `rows` and `cols` arguments; the default is 32768 x 256. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/row_sums.py`: the same three kernels with `cuda.shared.array`. `run(kernel, a)` copies `a` in, launches one 32-thread block per 32 rows and copies the sums back.
- `triton/row_sums.py`: one kernel that loads a block with `tl.load` and reduces it with `tl.sum`. Triton has no shared memory arrays to pad; the compiler chooses shared memory layouts, including swizzles, itself.
- `test_correctness.py`: checks every version against NumPy for shapes 1 x 32, 45 x 64, 100 x 96 and 4096 x 256. The inputs are small integers, so every sum is exact.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/row_sums.py
python triton/row_sums.py
pytest
```

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` and skips the CUDA test when `nvcc` is missing. The simulator does not model banks, so it checks the results only, not the conflicts.
