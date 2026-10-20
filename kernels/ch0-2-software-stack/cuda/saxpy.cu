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

__global__ void saxpy(float a, const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

int main() {
    const int n = 1000;
    const float a = 2.0f;
    size_t bytes = n * sizeof(float);

    std::vector<float> x(n), y(n), out(n);
    for (int i = 0; i < n; i++) {
        x[i] = float(i);
        y[i] = 1.0f;
    }

    float *d_x, *d_y, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(a, d_x, d_y, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        // The GPU may fuse a * x + y into one FMA, so compare with a tolerance.
        if (std::fabs(out[i] - (a * x[i] + y[i])) > 1e-5f * std::fabs(a * x[i] + y[i])) {
            std::printf("FAIL at %d: %f\n", i, out[i]);
            return 1;
        }
    }
    std::printf("OK\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
