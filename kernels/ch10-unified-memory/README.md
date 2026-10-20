# Chapter 10: Unified and pinned memory

`x = 2 * x`, with the CPU writing `x` before the kernel and reading it after, done with each kind of host memory.

- `cuda/unified_memory.cu`: the kernel `scale(x, n)` and five runs of it. Explicit `cudaMemcpy` from pageable (`malloc`) and from pinned (`cudaMallocHost`) memory; `cudaMallocManaged` with on-demand page faults; `cudaMallocManaged` with `cudaMemPrefetchAsync` to the GPU and back (CUDA 13 form, with a `cudaMemLocation`); and plain `malloc` memory passed to the kernel with a `cudaMemAdvise` preferred-location hint. The prefetch run needs `concurrentManagedAccess`, the `malloc` run needs `pageableMemoryAccess` (HMM or ATS); the program prints both attributes and skips a run its system cannot do. It exits with 1 on a mismatch. Needs CUDA 13 or later.
- `numba/unified_memory.py`: the same kernel with `cuda.to_device` from a NumPy array, `cuda.pinned_array`, `cuda.mapped_array` and `cuda.managed_array`. Numba has no wrapper for `cudaMemPrefetchAsync` or `cudaMemAdvise`.
- `test_correctness.py`: checks every version against NumPy for n = 1, 1000 and 4096. Under the Numba CUDA simulator the managed test is skipped, because the simulator has no `managed_array`. The CUDA test is skipped without `nvcc`.

There is no Triton version: Triton kernels take device tensors, and the host memory kind is PyTorch's business.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/unified_memory.py
pytest
```

Both programs take an optional element count. The default is 2<sup>26</sup> floats (256 MiB).

## Reading the output

Each line is one run: the time for data in, kernel, data out and the CPU loop that checks the result, and the bytes moved across the link (2 x the array) divided by that time. Compare the lines with each other and with your link's peak. The CPU loop is timed in every run because in the managed on-demand run that loop is where the pages fault back. The Python times include allocation and Python overhead.

Under `NUMBA_ENABLE_CUDASIM=1` the kernel runs on the CPU and nothing crosses a link, so the times mean nothing.
