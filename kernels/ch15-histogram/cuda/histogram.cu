#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "histogram_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

using Kernel = void (*)(const unsigned char *, int, unsigned int *);

const int REPS = 20;
const int THREADS = 256;

// Fills data with one of the three inputs the chapter uses.
static void make_input(std::vector<unsigned char> &data, const char *dist) {
    unsigned int state = 12345u;
    for (auto &x : data) {
        // xorshift32: a fixed sequence, so every run sees the same bytes.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        if (std::strcmp(dist, "uniform") == 0) {
            x = (unsigned char)(state & 255u);
        } else if (std::strcmp(dist, "skewed") == 0) {
            // Bin b has probability 2^-(b+1): half the bytes are 0, a quarter are 1, ...
            int b = 0;
            while (b < 31 && (state >> b & 1u)) ++b;
            x = (unsigned char)b;
        } else {
            x = 7;
        }
    }
}

// Runs one kernel REPS times and returns the mean time per run in ms.
static float time_kernel(Kernel k, int blocks, const unsigned char *d_data, int n, unsigned int *d_hist) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total = 0.0f;
    for (int r = 0; r <= REPS; ++r) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventRecord(start));
        k<<<blocks, THREADS>>>(d_data, n, d_hist);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        // The first launch loads the kernel onto the GPU, so it is not timed.
        if (r > 0) total += ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / REPS;
}

int main(int argc, char **argv) {
    long long requested = argc > 1 ? std::atoll(argv[1]) : 1 << 24;
    // The kernels index with int.
    if (requested < 1 || requested > INT_MAX) {
        std::fprintf(stderr, "need 1 <= n <= %d\n", INT_MAX);
        return 1;
    }
    int n = (int)requested;

    int device, sms;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    // A few blocks per SM; the grid-stride loop covers the rest of the input.
    int blocks = sms * 8;

    std::vector<unsigned char> h_data(n);
    std::vector<unsigned int> h_hist(BINS), ref(BINS);
    unsigned char *d_data;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));

    struct {
        const char *name;
        Kernel kernel;
    } variants[] = {
        {"global", hist_global},
        {"shared", hist_shared},
    };

    std::printf("%d bytes, %d blocks of %d threads\n", n, blocks, THREADS);
    int failures = 0;
    for (const char *dist : {"uniform", "skewed", "one-bin"}) {
        make_input(h_data, dist);
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n, cudaMemcpyHostToDevice));
        std::fill(ref.begin(), ref.end(), 0u);
        for (unsigned char x : h_data) ++ref[x];
        for (auto &v : variants) {
            float ms = time_kernel(v.kernel, blocks, d_data, n, d_hist);
            CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            bool ok = h_hist == ref;
            failures += !ok;
            std::printf("  %-8s %-7s %9.3f ms  %8.1f GB/s%s\n", dist, v.name, ms, n / (ms * 1e-3) / 1e9,
                        ok ? "" : "  MISMATCH");
        }
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
