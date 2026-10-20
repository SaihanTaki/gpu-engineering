#include <climits>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "reduce_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

using Kernel = void (*)(const float *, float *, unsigned);

const int THREADS = 256;
const int REPS = 20;

// Launches k on blocks of THREADS until one value is left, ping-ponging
// between two buffers of partial sums. Returns a pointer to that value.
static float *multi_pass(Kernel k, const float *d_in, float *d_a, float *d_b, unsigned n) {
    const float *src = d_in;
    float *dst = d_a;
    do {
        unsigned blocks = (n + THREADS - 1) / THREADS;
        k<<<blocks, THREADS, THREADS * sizeof(float)>>>(src, dst, n);
        CUDA_CHECK(cudaGetLastError());
        src = dst;
        dst = dst == d_a ? d_b : d_a;
        n = blocks;
    } while (n > 1);
    return const_cast<float *>(src);
}

static float *one_pass_atomic(const float *d_in, float *d_out, unsigned n, int blocks) {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
    reduce_atomic<<<blocks, THREADS>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    return d_out;
}

int main(int argc, char **argv) {
    long long arg = argc > 1 ? std::atoll(argv[1]) : 1 << 22;
    if (arg < 1 || arg > INT_MAX) {
        std::fprintf(stderr, "need 1 <= n <= %d\n", INT_MAX);
        return 1;
    }
    unsigned n = (unsigned)arg;

    // Whole numbers from -3 to 3, so a float32 sum is exact while every
    // partial sum stays below 2^24, and order cannot change the result.
    std::vector<float> h_in(n);
    long long expected = 0;
    for (unsigned i = 0; i < n; ++i) {
        int v = (int)(i * 2654435761u % 7u) - 3;
        h_in[i] = (float)v;
        expected += v;
    }

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int atomic_blocks = 4 * sms;

    unsigned partials = (n + THREADS - 1) / THREADS;
    float *d_in, *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, partials * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, partials * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    struct {
        const char *name;
        Kernel kernel;
    } variants[] = {
        {"interleaved", reduce_interleaved},
        {"sequential", reduce_sequential},
        {"warp_final", reduce_warp_final},
        {"atomic", nullptr},
    };

    // Bandwidth counts each input element read once.
    std::printf("n = %u floats, %d threads per block, expected sum %lld\n", n, THREADS, expected);
    int failures = 0;
    for (auto &v : variants) {
        auto run = [&]() {
            return v.kernel ? multi_pass(v.kernel, d_in, d_a, d_b, n) : one_pass_atomic(d_in, d_a, n, atomic_blocks);
        };
        // The first run loads the kernels onto the GPU, so it is not timed.
        float *d_result = run();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < REPS; ++r) {
            d_result = run();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= REPS;
        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = (long long)got == expected;
        failures += !ok;
        std::printf("  %-11s %9.3f ms  %8.1f GB/s  sum %.0f%s\n", v.name, ms, n * sizeof(float) / (ms * 1e-3) / 1e9,
                    got, ok ? "" : "  MISMATCH");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
