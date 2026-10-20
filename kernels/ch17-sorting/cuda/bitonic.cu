#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "bitonic_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// Every stage is its own launch: the launch boundary is the barrier between stages.
static void sort_global(float *d, int m) {
    int blocks = (m / 2 + THREADS - 1) / THREADS;
    for (int k = 2; k <= m; k *= 2)
        for (int j = k / 2; j > 0; j /= 2)
            bitonic_step<<<blocks, THREADS>>>(d, m, j, k);
    CUDA_CHECK(cudaGetLastError());
}

// Stages with j < TILE run inside shared memory; only j >= TILE needs a global launch. Needs m >= TILE.
static void sort_shared(float *d, int m) {
    int tiles = m / TILE;
    bitonic_shared<<<tiles, TILE_THREADS>>>(d, 2, TILE);
    for (int k = 2 * TILE; k <= m; k *= 2) {
        for (int j = k / 2; j >= TILE; j /= 2)
            bitonic_step<<<(m / 2 + THREADS - 1) / THREADS, THREADS>>>(d, m, j, k);
        bitonic_shared<<<tiles, TILE_THREADS>>>(d, k, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void sort_thrust(float *d, int m) { thrust::sort(thrust::device, d, d + m); }

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
    if (n < 1 || n > (1 << 30)) {
        std::fprintf(stderr, "need 1 <= n <= 2^30\n");
        return 1;
    }
    // Pad to a power of two (at least one tile) with +inf, which sorts to the end.
    int m = TILE;
    while (m < n) m *= 2;

    std::vector<float> h(n), out(m), ref(n);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : h) x = dist(gen);
    ref = h;
    std::sort(ref.begin(), ref.end());

    std::vector<float> padded(m, INFINITY);
    std::copy(h.begin(), h.end(), padded.begin());
    float *d;
    CUDA_CHECK(cudaMalloc(&d, m * sizeof(float)));

    struct {
        const char *name;
        void (*sort)(float *, int);
    } variants[] = {{"global", sort_global}, {"shared", sort_shared}, {"thrust", sort_thrust}};

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("%d keys, padded to %d\n", n, m);
    int failures = 0;
    for (auto &v : variants) {
        // The first run loads the kernels, so time a second run on fresh input.
        for (int rep = 0; rep < 2; ++rep) {
            CUDA_CHECK(cudaMemcpy(d, padded.data(), m * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            v.sort(d, m);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(out.data(), d, m * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::equal(ref.begin(), ref.end(), out.begin());
        failures += !ok;
        std::printf("  %-7s %9.3f ms  %8.1f Mkeys/s%s\n", v.name, ms, n / (ms * 1e3), ok ? "" : "  MISMATCH");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d));
    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
