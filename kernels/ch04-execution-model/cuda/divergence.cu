#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
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

// A multiple of 32, so every warp of a block is full.
constexpr int THREADS = 256;

__global__ void branchy(const float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Lanes of one warp with different signs take different paths.
        if (x[i] > 0.0f) {
            out[i] = x[i] * 2.0f + 1.0f;
        } else {
            out[i] = x[i] * 0.5f;
        }
    }
}

__global__ void branch_free(const float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // Every lane runs the same statements and keeps one of the two results.
        float a = v * 2.0f + 1.0f;
        float b = v * 0.5f;
        out[i] = v > 0.0f ? a : b;
    }
}

__global__ void warp_masks(const float *x, unsigned *masks, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool take_if = i < n && x[i] > 0.0f;
    // Bit k is set when lane k of this warp would take the if path.
    unsigned mask = __ballot_sync(0xffffffffu, take_if);
    if (threadIdx.x % 32 == 0 && i < n) {
        masks[i / 32] = mask;
    }
}

static float reference(float v) { return v > 0.0f ? v * 2.0f + 1.0f : v * 0.5f; }

// Runs all three kernels on x and checks them against the CPU. Returns the mismatch count.
static int check(const std::vector<float> &x, const char *label) {
    int n = static_cast<int>(x.size());
    int warps = (n + 31) / 32;
    int blocks = (n + THREADS - 1) / THREADS;

    float *d_x, *d_out;
    unsigned *d_masks;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_masks, warps * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> out(n);
    std::vector<unsigned> masks(warps);
    int errors = 0;

    branchy<<<blocks, THREADS>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) errors += out[i] != reference(x[i]);

    branch_free<<<blocks, THREADS>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) errors += out[i] != reference(x[i]);

    warp_masks<<<blocks, THREADS>>>(d_x, d_masks, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(masks.data(), d_masks, warps * sizeof(unsigned), cudaMemcpyDeviceToHost));
    int divergent = 0;
    for (int w = 0; w < warps; ++w) {
        unsigned expected = 0;
        for (int lane = 0; lane < 32 && w * 32 + lane < n; ++lane) {
            if (x[w * 32 + lane] > 0.0f) expected |= 1u << lane;
        }
        errors += masks[w] != expected;
        // A partial last warp counts as uniform when all of its real lanes agree.
        int lanes = std::min(32, n - w * 32);
        unsigned all = lanes == 32 ? 0xffffffffu : (1u << lanes) - 1;
        divergent += masks[w] != 0 && masks[w] != all;
    }
    std::printf("%s: %d of %d warps diverge\n", label, divergent, warps);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_masks));
    return errors;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    std::vector<float> x(n);
    std::mt19937 gen(0);
    std::normal_distribution<float> normal;
    for (float &v : x) v = normal(gen);

    int errors = check(x, "random signs");
    std::sort(x.begin(), x.end());
    errors += check(x, "sorted      ");

    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
