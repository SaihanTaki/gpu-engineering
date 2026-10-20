# Course Docker image

One image with what you need to build, run, test and profile the course
kernels on a local NVIDIA GPU: CUDA 13.2 (`nvcc`, `compute-sanitizer`),
Nsight Compute (`ncu`), Nsight Systems (`nsys`), CMake, PyTorch, Triton,
`numba-cuda`, NumPy, pytest and JupyterLab. Python versions are pinned in
[`requirements.txt`](requirements.txt).

## Host requirements

- An NVIDIA GPU with a driver recent enough for CUDA 13.2.
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `--gpus all` works.

## Build

From the repository root:

```bash
docker build -t gpu-engineering docker/
```

CI publishes the same image to `ghcr.io/OWNER/gpu-engineering` on every
release tag. Replace `OWNER` with the GitHub user or organization that hosts
the repository, then pull it instead of building:

```bash
docker pull ghcr.io/OWNER/gpu-engineering:latest
```

## Run

From the repository root:

```bash
docker run --gpus all -it --rm -p 8888:8888 -v "$PWD":/workspace gpu-engineering
```

This starts JupyterLab on port 8888. Open the URL with the token that it
prints. To get a shell instead, add `bash` to the end of the command.

## Inside the container

```bash
pytest kernels                          # every kernel, all three languages
pytest kernels/ch06-coalescing          # one chapter
make -C kernels/ch06-coalescing/cuda all  # build one chapter's CUDA C++ code
```

## Profiling

Nsight Compute reads GPU performance counters, which most drivers restrict to
admin users by default. Either start the container with
`--cap-add=SYS_ADMIN`:

```bash
docker run --gpus all --cap-add=SYS_ADMIN -it --rm -v "$PWD":/workspace gpu-engineering bash
```

or allow all users to read the counters on the host by setting the driver
option `NVreg_RestrictProfilingOnAdminUsers=0` (see NVIDIA's
[ERR_NVGPUCTRPERM guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)).
