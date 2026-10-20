#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "conv2d_kernels.cuh"

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

// The same sum as the kernels, in double, with zero padding.
static std::vector<float> reference(const std::vector<float> &in, const std::vector<float> &f, int R, int h, int w) {
    std::vector<float> out((size_t)h * w);
    int k = 2 * R + 1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w)
                        sum += (double)f[(dy + R) * k + dx + R] * in[(size_t)yy * w + xx];
                }
            out[(size_t)y * w + x] = (float)sum;
        }
    return out;
}

// Runs one kernel REPS times and returns the mean time per run in ms.
static float time_kernel(Kernel kernel, const float *d_in, float *d_out, int h, int w) {
    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    // The first launch loads the kernel onto the GPU, so it is not timed.
    kernel<<<grid, block>>>(d_in, d_out, h, w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) kernel<<<grid, block>>>(d_in, d_out, h, w);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

int main(int argc, char **argv) {
    int h = argc > 1 ? std::atoi(argv[1]) : 2048;
    int w = argc > 2 ? std::atoi(argv[2]) : h;
    // The kernels index with int.
    if (h < 1 || w < 1 || (long long)h * w > INT_MAX) {
        std::fprintf(stderr, "need 1 <= h * w <= %d\n", INT_MAX);
        return 1;
    }
    size_t n = (size_t)h * w;
    size_t bytes = n * sizeof(float);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> h_in(n), h_out(n);
    for (auto &v : h_in) v = uniform(rng);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    struct {
        int R;
        const char *name;
        Kernel kernel;
    } variants[] = {
        {1, "naive", conv2d_naive<1>}, {1, "tiled", conv2d_tiled<1>},
        {2, "naive", conv2d_naive<2>}, {2, "tiled", conv2d_tiled<2>},
        {3, "naive", conv2d_naive<3>}, {3, "tiled", conv2d_tiled<3>},
    };

    std::printf("%d x %d floats, %d x %d threads per block\n", h, w, TILE, TILE);
    int failures = 0;
    int loaded_r = -1;
    std::vector<float> want;
    for (auto &v : variants) {
        int k = 2 * v.R + 1;
        if (v.R != loaded_r) {
            std::vector<float> f(k * k);
            for (auto &x : f) x = uniform(rng);
            CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.data(), f.size() * sizeof(float)));
            want = reference(h_in, f, v.R, h, w);
            loaded_r = v.R;
        }
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel(v.kernel, d_in, d_out, h, w);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        size_t errors = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::fabs(h_out[i] - want[i]) > 1e-4f * (1.0f + std::fabs(want[i]))) ++errors;
        failures += errors != 0;
        // Effective bandwidth counts each pixel read once and written once.
        std::printf("  R=%d %dx%d %-6s %9.3f ms  %8.1f GB/s%s\n", v.R, k, k, v.name, ms,
                    2.0 * bytes / (ms * 1e-3) / 1e9, errors ? "  MISMATCH" : "");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
