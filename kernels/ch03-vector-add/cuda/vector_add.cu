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

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The last block can have threads past the end of the arrays.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static double gb_per_s(size_t bytes, float ms) {
    return bytes / (ms * 1e-3) / 1e9;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);

    // 1. Fill the host arrays.
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (i % 1000) * 0.001f;
        h_b[i] = 1.0f - (i % 7) * 0.25f;
    }

    // 2. Allocate the device arrays.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // The first launch loads the kernel onto the GPU, so it is not timed.
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, copied_in, computed, copied_out;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copied_in));
    CUDA_CHECK(cudaEventCreate(&computed));
    CUDA_CHECK(cudaEventCreate(&copied_out));

    // 3. Copy a and b from host to device.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(copied_in));

    // 4. Launch: 256 threads per block, enough blocks to cover n.
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    // 5. Launch errors show up here, errors inside the kernel at the sync.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(computed));

    // 6. Copy c from device to host.
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(copied_out));
    CUDA_CHECK(cudaEventSynchronize(copied_out));

    float in_ms = 0.0f, kernel_ms = 0.0f, out_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&in_ms, start, copied_in));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, copied_in, computed));
    CUDA_CHECK(cudaEventElapsedTime(&out_ms, computed, copied_out));

    // 7. Check against a CPU loop.
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            ++errors;
        }
    }

    // 8. Free the device arrays.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(copied_in));
    CUDA_CHECK(cudaEventDestroy(computed));
    CUDA_CHECK(cudaEventDestroy(copied_out));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Bytes moved per phase: copy in a and b, kernel reads a, b and writes c, copy out c.
    std::printf("n = %d, %d blocks of %d threads\n", n, blocks, threads);
    std::printf("  copy in  (a, b):  %9.3f ms  %8.1f GB/s\n", in_ms, gb_per_s(2 * bytes, in_ms));
    std::printf("  kernel:           %9.3f ms  %8.1f GB/s\n", kernel_ms, gb_per_s(3 * bytes, kernel_ms));
    std::printf("  copy out (c):     %9.3f ms  %8.1f GB/s\n", out_ms, gb_per_s(bytes, out_ms));

    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
