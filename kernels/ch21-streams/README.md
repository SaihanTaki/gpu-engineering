# Chapter 21: Streams and concurrency

Copy `n` floats to the GPU, run `iters` steps of `v = 0.999 v + 0.001` on each, copy them back. Done serially, and as a chunked pipeline where chunk `c` goes to stream `c % streams`.

- `cuda/streams_kernels.cuh`: the kernel `work(x, n, iters)`. It uses `__fmul_rn` and `__fadd_rn` so the GPU result matches a CPU loop bit for bit. The header holds only device code, so NVRTC can compile it without `nvcc`.
- `cuda/streams.cu`: prints `asyncEngineCount`, then times four runs with `cudaEvent` over 10 repetitions: serial in the default stream; the pipeline from pinned memory (`cudaMallocHost`); the same pipeline from pageable memory (`std::vector`); and the pinned pipeline captured once into a CUDA Graph with `cudaStreamBeginCapture` and replayed with `cudaGraphLaunch`. Every result is compared with the serial one, which is compared with a CPU loop. Exits with 1 on a mismatch. The Makefile builds for `sm_80` (A100); set `ARCH=75` for a T4 or `ARCH=90` for an H100. It passes `-ffp-contract=off` to the host compiler so the CPU check does not fuse into an FMA.
- `numba/streams.py`: the serial run and the pipeline with `cuda.stream()`, `cuda.pinned_array`, and `copy_to_device` / `copy_to_host` with `stream=` on slices of one device array. One run is timed with `cuda.event()` pairs joined across the streams; the comparison times use `time.perf_counter()`, so they include Python's launch cost.
- `test_correctness.py`: checks the Numba pipeline against the serial run for several stream and chunk counts (a ragged last chunk, more chunks than elements), and the serial run against NumPy. It compiles the CUDA kernel with NVRTC for `compute_75`, `compute_80` and `compute_90` and checks the PTX has no FMA, and compiles the Numba kernel to PTX. The CUDA program test is skipped without `nvcc`.

There is no Triton version: Triton kernels launch into PyTorch's current stream (`with torch.cuda.stream(s):`), and the copies belong to PyTorch.

Run inside the course Docker image, from this folder:

```sh
make -C cuda run
python numba/streams.py
pytest
```

Both programs take `[n] [streams] [chunks] [iters]`. The defaults are 2<sup>24</sup> floats (64 MiB), 4 streams, 16 chunks, 100 iterations.

## Reading the output

Each line is the mean time of one run and its speedup over the serial run. The pinned pipeline should beat serial when the kernel and copies take similar time; the pageable one should not. The graph line differs from the stream line by the host's launch cost, which matters most with many small chunks (try `./cuda/streams 65536 4 64 10`).

Under `NUMBA_ENABLE_CUDASIM=1` the Numba simulator runs every call synchronously on the CPU. The results are checked, but nothing overlaps, events return -1, and the times mean nothing.
