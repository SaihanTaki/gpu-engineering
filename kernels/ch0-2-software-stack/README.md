# Chapter 0.2: The GPU software stack

One SAXPY kernel (`out = a * x + y`) in three languages, and every stage of its compilation for sm_80.

- `cuda/saxpy.cu`: the kernel and a `main()` that checks it. The Makefile has `ptx`, `sass` and `fatbin` targets that print each stage with `nvcc` and `cuobjdump`. Set `ARCH=75` for a T4 or `ARCH=90` for an H100.
- `numba/saxpy.py`, `triton/saxpy.py`: the same kernel with `@cuda.jit` and `@triton.jit`.
- `generate.py`: compiles the Numba and Triton kernels for sm_80 without a GPU and writes every stage to `generated/` (NVVM IR, TTIR, TTGIR, LLVM IR, PTX, cubin and SASS). Tool versions are in `generated/VERSIONS.txt`.
- `test_correctness.py`: checks the Numba and Triton kernels against NumPy, runs the CUDA binary when `nvcc` is present, and checks that the generated files exist.

The chapter's simulation shows the files in `generated/` as they are. After changing a kernel or a pinned version, regenerate and commit them:

```sh
python generate.py
```

Inside the course Docker image, from this folder:

```sh
make -C cuda run
make -C cuda ptx
make -C cuda sass
make -C cuda fatbin
pytest
```

Without an NVIDIA GPU, `pytest` runs the Numba and Triton kernels on the CPU (`NUMBA_ENABLE_CUDASIM=1`, `TRITON_INTERPRET=1`) and skips the CUDA test when `nvcc` is missing.
