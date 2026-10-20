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

__global__ void whoami(int *out, int *block, int *thread, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The last block can have threads past the end of the array.
    if (i < n) {
        out[i] = i;
        block[i] = blockIdx.x;
        thread[i] = threadIdx.x;
    }
}

__global__ void whoami2d(int *out, int width, int height) {
    // x is the fast-moving dimension, so it maps to columns.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        out[row * width + col] = row * width + col;
    }
}

static int run1d(int n, int threads) {
    int blocks = (n + threads - 1) / threads;
    size_t bytes = n * sizeof(int);

    int *d_out, *d_block, *d_thread;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_block, bytes));
    CUDA_CHECK(cudaMalloc(&d_thread, bytes));

    whoami<<<blocks, threads>>>(d_out, d_block, d_thread, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> out(n), block(n), thread(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(block.data(), d_block, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(thread.data(), d_thread, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_block));
    CUDA_CHECK(cudaFree(d_thread));

    std::printf("%5s %6s %7s\n", "i", "block", "thread");
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        std::printf("%5d %6d %7d\n", out[i], block[i], thread[i]);
        if (out[i] != i || block[i] != i / threads || thread[i] != i % threads) {
            ++errors;
        }
    }
    return errors;
}

static int run2d(int width, int height) {
    dim3 block(4, 2);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t bytes = width * height * sizeof(int);

    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    whoami2d<<<grid, block>>>(d_out, width, height);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> out(width * height);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    int errors = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (out[row * width + col] != row * width + col) {
                ++errors;
            }
        }
    }
    return errors;
}

int main() {
    int errors = run1d(20, 8) + run2d(5, 3);
    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
