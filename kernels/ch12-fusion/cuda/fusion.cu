#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "fusion_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

const int THREADS = 256;
const int REPS = 20;
const float A = 1.5f;

__global__ void empty() {}

struct Buffers {
    float *x, *y, *t, *u, *out;
    int n;
};

static void run_unfused(const Buffers &b) {
    int blocks = (b.n + THREADS - 1) / THREADS;
    scale<<<blocks, THREADS>>>(b.x, b.t, A, b.n);
    add<<<blocks, THREADS>>>(b.t, b.y, b.u, b.n);
    relu<<<blocks, THREADS>>>(b.u, b.out, b.n);
}

static void run_fused(const Buffers &b) {
    int blocks = (b.n + THREADS - 1) / THREADS;
    fused<<<blocks, THREADS>>>(b.x, b.y, b.out, A, b.n);
}

static void run_empty(const Buffers &) { empty<<<1, 1>>>(); }

// Mean time of one call of `run` over REPS calls, in ms.
static float time_ms(void (*run)(const Buffers &), const Buffers &b) {
    // The first call loads the kernels onto the GPU, so it is not timed.
    run(b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) run(b);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

static int check(const std::vector<float> &got, const std::vector<float> &x, const std::vector<float> &y) {
    int errors = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float want = std::fmax(A * x[i] + y[i], 0.0f);
        // The fused kernel may contract a * x + y into one FMA, which rounds once instead of twice.
        if (std::fabs(got[i] - want) > 1e-5f * (1.0f + std::fabs(want))) ++errors;
    }
    return errors;
}

int main(int argc, char **argv) {
    long long nn = argc > 1 ? std::atoll(argv[1]) : (1 << 24);
    if (nn < 1 || nn > INT_MAX / 4) {
        std::fprintf(stderr, "need 1 <= n <= %d\n", INT_MAX / 4);
        return 1;
    }
    int n = (int)nn;
    size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> h_x(n), h_y(n), h_out(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = (float)(i % 1000) / 500.0f - 1.0f;
        h_y[i] = (float)(i % 777) / 777.0f - 0.5f;
    }

    Buffers b{};
    b.n = n;
    CUDA_CHECK(cudaMalloc(&b.x, bytes));
    CUDA_CHECK(cudaMalloc(&b.y, bytes));
    CUDA_CHECK(cudaMalloc(&b.t, bytes));
    CUDA_CHECK(cudaMalloc(&b.u, bytes));
    CUDA_CHECK(cudaMalloc(&b.out, bytes));
    CUDA_CHECK(cudaMemcpy(b.x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    struct {
        const char *name;
        void (*run)(const Buffers &);
        int arrays;  // arrays read plus arrays written, each n floats
    } variants[] = {{"unfused", run_unfused, 7}, {"fused", run_fused, 3}};

    std::printf("n = %d floats, %d threads per block\n", n, THREADS);
    int failures = 0;
    for (auto &v : variants) {
        CUDA_CHECK(cudaMemset(b.out, 0, bytes));
        float ms = time_ms(v.run, b);
        CUDA_CHECK(cudaMemcpy(h_out.data(), b.out, bytes, cudaMemcpyDeviceToHost));
        int errors = check(h_out, h_x, h_y);
        failures += errors != 0;
        double moved = (double)v.arrays * bytes;
        std::printf("  %-8s %9.4f ms  %7.1f MB moved  %8.1f GB/s%s\n", v.name, ms, moved / 1e6,
                    moved / (ms * 1e-3) / 1e9, errors ? "  MISMATCH" : "");
    }
    // Back-to-back launches of a kernel that does nothing: the fixed cost per launch on this machine.
    std::printf("  %-8s %9.2f us per launch\n", "empty", 1e3f * time_ms(run_empty, b));

    CUDA_CHECK(cudaFree(b.x));
    CUDA_CHECK(cudaFree(b.y));
    CUDA_CHECK(cudaFree(b.t));
    CUDA_CHECK(cudaFree(b.u));
    CUDA_CHECK(cudaFree(b.out));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
