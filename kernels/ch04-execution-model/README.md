# Chapter 4: Execution model

The same elementwise function written two ways, `out = x > 0 ? 2x + 1 : x / 2`: once with an `if/else` that splits a warp when its lanes disagree, and once branch-free, where every lane runs the same statements and keeps one result.

- `cuda/divergence.cu`: `branchy` (if/else), `branch_free` (compute both, select) and `warp_masks`, which records with `__ballot_sync` which lanes of each warp take the `if` path. `main()` runs all three on normally distributed data, checks them against a CPU loop, prints how many warps diverge, then sorts the data and does it again. It exits with 1 on a mismatch. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/divergence.py`: `branchy` and `branch_free` with `cuda.grid(1)`. `run(kernel, x)` copies `x` in, launches 256 threads per block and copies the result back. The ballot kernel is left out because the Numba CUDA simulator has no `ballot_sync`.
- `triton/divergence.py`: one kernel that uses `tl.where`. Triton has no per-lane branches; an `if` in a Triton kernel needs a scalar condition.
- `test_correctness.py`: checks every version against NumPy for n = 1, 31, 1000 and 2<sup>20</sup>, with random and with sorted inputs.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/divergence.py
python triton/divergence.py
pytest
```

`./cuda/divergence` takes an optional element count; the default is 2<sup>20</sup>. With random signs nearly every warp diverges. After sorting, at most one warp diverges: the one where the values change sign.

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` and skips the CUDA test when `nvcc` is missing.
