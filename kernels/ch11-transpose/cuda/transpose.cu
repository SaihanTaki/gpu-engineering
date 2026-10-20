#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "transpose_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

using Kernel = void (*)(const float *, float *, int, int);

const int REPS = 20;

// Runs one kernel REPS times and returns the mean time per run in ms.
static float time_kernel(Kernel k, const float *d_in, float *d_out, int rows, int cols) {
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    // The first launch loads the kernel onto the GPU, so it is not timed.
    k<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) {
        k<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    int cols = argc > 2 ? std::atoi(argv[2]) : rows;
    // The kernels index with int.
    if (rows < 1 || cols < 1 || (long long)rows * cols > INT_MAX) {
        std::fprintf(stderr, "need 1 <= rows * cols <= %d\n", INT_MAX);
        return 1;
    }
    size_t n = (size_t)rows * cols;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_in(n), h_out(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = (float)(i % 1000003);
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    struct {
        const char *name;
        Kernel kernel;
    } variants[] = {
        {"naive", transpose_naive},
        {"tiled", transpose_shared<0>},
        {"padded", transpose_shared<1>},
    };

    // Effective bandwidth counts each element read once and written once.
    std::printf("%d x %d floats, %d x %d threads per block\n", rows, cols, TILE, BLOCK_ROWS);
    int failures = 0;
    for (auto &v : variants) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel(v.kernel, d_in, d_out, rows, cols);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (h_out[(size_t)c * rows + r] != h_in[(size_t)r * cols + c]) {
                    ++errors;
                }
            }
        }
        failures += errors != 0;
        std::printf("  %-7s %9.3f ms  %8.1f GB/s%s\n", v.name, ms, 2.0 * bytes / (ms * 1e-3) / 1e9,
                    errors ? "  MISMATCH" : "");
    }

    // A device-to-device copy moves the same bytes with no transpose: the ceiling to compare with.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) {
        CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float copy_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start, stop));
    copy_ms /= REPS;
    std::printf("  %-7s %9.3f ms  %8.1f GB/s\n", "memcpy", copy_ms, 2.0 * bytes / (copy_ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
