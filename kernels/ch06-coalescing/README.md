# Chapter 6: Global memory and coalescing

Copy kernels whose only difference is the address pattern of their loads: an offset copy, a strided copy, and reading one field of a particle stored as an array of structs (AoS) or a struct of arrays (SoA).

- `cuda/coalescing.cu`: `copy_offset` (`out[i] = in[i + offset]`), `copy_strided` (`out[i] = in[i * stride]`), `get_x_aos` (reads `p[i].x` of a 12-byte `Particle`) and `get_x_soa` (reads `x[i]`). `main()` runs each one, checks it against a CPU loop, and prints its time and useful bandwidth: bytes the threads read and wrote, per second. It exits with 1 on a mismatch. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/coalescing.py`: the same four kernels. The AoS array is an `n x 3` float32 array, one row per particle, which has the same memory layout as the CUDA struct.
- `triton/coalescing.py`: a strided copy written with pointer arithmetic, and the AoS field read written with `tl.make_block_ptr`. Triton 3.8 marks `make_block_ptr` deprecated; it still works, and the wrapper silences the warning.
- `test_correctness.py`: checks every version against NumPy for n = 1, 1000 and 2<sup>18</sup>, offsets 0, 1, 8 and 31, and strides 1, 2, 3, 8 and 32.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/coalescing.py
python triton/coalescing.py
pytest
```

`./cuda/coalescing` takes an optional element count; the default is 2<sup>20</sup>. It allocates 32 floats of input per element, 128 MB at the default, so the stride-32 copy has data to read.

Without a GPU, `pytest` sets `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` and skips the CUDA test when `nvcc` is missing.
