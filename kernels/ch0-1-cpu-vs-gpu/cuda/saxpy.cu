#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

void saxpy_cpu(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The last block can have threads past the end of the arrays.
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

static int run(int n, int threads) {
    const float a = 2.0f;
    std::vector<float> x(n), y(n), expected(n), result(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 1000) * 0.001f;
        y[i] = 1.0f - (i % 7) * 0.25f;
    }
    expected = y;

    auto t0 = std::chrono::steady_clock::now();
    saxpy_cpu(n, a, x.data(), expected.data());
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t bytes = n * sizeof(float);
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemset(d_x, 0, bytes));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes));

    cudaEvent_t start, copied, done;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copied));
    CUDA_CHECK(cudaEventCreate(&done));

    int blocks = (n + threads - 1) / threads;
    // The first launch pays one-time setup costs, so it is not timed.
    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(copied));
    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done));
    CUDA_CHECK(cudaEventSynchronize(done));

    float copy_ms = 0.0f, kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start, copied));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, copied, done));

    CUDA_CHECK(cudaMemcpy(result.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(copied));
    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        // The compiler may fuse a * x + y into one FMA, which rounds once instead of twice.
        if (std::fabs(result[i] - expected[i]) > 1e-6f * std::fmax(1.0f, std::fabs(expected[i]))) {
            ++errors;
        }
    }

    std::printf("n = %d, %d blocks of %d threads\n", n, blocks, threads);
    std::printf("  CPU loop:            %9.3f ms\n", cpu_ms);
    std::printf("  GPU copy x, y in:    %9.3f ms\n", copy_ms);
    std::printf("  GPU kernel:          %9.3f ms\n", kernel_ms);
    return errors;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : (1 << 24) + 3;
    int errors = run(1000, 256) + run(n, 256);
    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
