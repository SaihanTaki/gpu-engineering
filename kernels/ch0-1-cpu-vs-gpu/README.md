# Chapter 0.1: CPU vs GPU

SAXPY (`y = a * x + y`) on the CPU and on the GPU, timed side by side.

- `cuda/saxpy.cu`: a CPU loop and a GPU kernel with a rounded-up grid and a bounds check. `main()` checks the GPU result against the CPU loop and prints the CPU time (`std::chrono`) and the copy and kernel times (`cudaEvent`). It exits with 1 on a mismatch.
- `numba/saxpy.py`: the same loop compiled for the CPU with `@njit`, and the kernel with `@cuda.jit`.
- `triton/saxpy.py`: the block-level kernel, compared with NumPy.
- `test_correctness.py`: checks every version against NumPy, including sizes that are not a multiple of the block size.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/saxpy.py
python triton/saxpy.py
pytest
```

Each script takes an optional element count, for example `python numba/saxpy.py 1000000`. The default is 2<sup>24</sup> + 3.

The times are only meaningful on a real GPU. Without one, set `NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` and pass a small count: the simulator and the interpreter run on the CPU and are far slower than either real version. `pytest` sets both variables for you when `nvidia-smi` is missing, and skips the CUDA test when `nvcc` is missing.
