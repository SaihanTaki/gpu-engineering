// Runs each bug next to its fix and checks both against the CPU.
//   ./bugs race|oob|nosync|uninit|divergent|launch|all
// The buggy result may look right without a tool; run it under compute-sanitizer.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "bugs_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// A launch returns no error code: launch errors wait in cudaGetLastError(),
// and errors while the kernel runs show up at the next synchronization.
#define CUDA_CHECK_LAUNCH()                     \
    do {                                        \
        CUDA_CHECK(cudaGetLastError());         \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while (0)

static int fixed_failures = 0;

static void report(const char *bug, const char *version, bool ok) {
    std::printf("  %-9s %-6s %s\n", bug, version, ok ? "OK" : "WRONG");
    if (!ok && std::strcmp(version, "fixed") == 0) ++fixed_failures;
}

static int blocks_for(int n) { return (n + BLOCK - 1) / BLOCK; }

template <typename T>
static T *to_device(const std::vector<T> &h) {
    T *d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> to_host(const T *d, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// Values in [-1, 1) from a fixed xorshift sequence, so every run sees the same data.
static std::vector<float> make_input(int n) {
    std::vector<float> x(n);
    unsigned int s = 12345u;
    for (auto &v : x) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v = (float)(s % 2000u) / 1000.0f - 1.0f;
    }
    return x;
}

static void run_race(int n) {
    auto x = make_input(n);
    unsigned int expected = 0;
    for (float v : x) expected += v > 0.0f;
    float *d_x = to_device(x);
    unsigned int *d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned int)));
    using Kernel = void (*)(const float *, int, unsigned int *);
    Kernel kernels[] = {count_racy, count_atomic};
    const char *names[] = {"buggy", "fixed"};
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned int)));
        kernels[k]<<<blocks_for(n), BLOCK>>>(d_x, n, d_total);
        CUDA_CHECK_LAUNCH();
        unsigned int got = to_host(d_total, 1)[0];
        std::printf("  race      %-6s counted %u of %u positives\n", names[k], got, expected);
        report("race", names[k], got == expected);
    }
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_total));
}

static void run_oob(int n) {
    auto x = make_input(n);
    float *d_in = to_device(x);
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    using Kernel = void (*)(const float *, int, float *);
    Kernel kernels[] = {diff_oob, diff};
    const char *names[] = {"buggy", "fixed"};
    for (int k = 0; k < 2; ++k) {
        kernels[k]<<<blocks_for(n), BLOCK>>>(d_in, n, d_out);
        CUDA_CHECK_LAUNCH();
        // Only the n - 1 defined outputs are compared, so the stray read leaves no trace here.
        auto out = to_host(d_out, n - 1);
        bool ok = true;
        for (int i = 0; i < n - 1; ++i) ok &= out[i] == x[i + 1] - x[i];
        report("oob", names[k], ok);
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

static void run_nosync(int n) {
    n = blocks_for(n) * BLOCK;
    auto x = make_input(n);
    float *d_in = to_device(x);
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    using Kernel = void (*)(const float *, float *);
    Kernel kernels[] = {reverse_nosync, reverse};
    const char *names[] = {"buggy", "fixed"};
    for (int k = 0; k < 2; ++k) {
        kernels[k]<<<n / BLOCK, BLOCK>>>(d_in, d_out);
        CUDA_CHECK_LAUNCH();
        auto out = to_host(d_out, n);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            int b = i / BLOCK, t = i % BLOCK;
            bad += out[i] != x[b * BLOCK + BLOCK - 1 - t];
        }
        std::printf("  nosync    %-6s %d of %d elements wrong\n", names[k], bad, n);
        report("nosync", names[k], bad == 0);
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

static void run_uninit(int n) {
    int *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(int)));
    const char *names[] = {"buggy", "fixed"};
    for (int k = 0; k < 2; ++k) {
        // Fresh memory often holds zeros, so the buggy version can pass by luck.
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(int)));
        // The bug: n bytes, not n ints, so 3/4 of v stays uninitialized.
        size_t bytes = k == 0 ? n : n * sizeof(int);
        CUDA_CHECK(cudaMemset(d_v, 0, bytes));
        add_index<<<blocks_for(n), BLOCK>>>(d_v, n);
        CUDA_CHECK_LAUNCH();
        auto v = to_host(d_v, n);
        bool ok = true;
        for (int i = 0; i < n; ++i) ok &= v[i] == i;
        report("uninit", names[k], ok);
    }
    CUDA_CHECK(cudaFree(d_v));
}

static void run_divergent(int n) {
    // 100 elements in the last block: its fourth warp holds both valid and invalid threads.
    n = (blocks_for(n) - 1) * BLOCK + 100;
    auto x = make_input(n);
    float *d_x = to_device(x);
    float *d_sums;
    int blocks = blocks_for(n);
    CUDA_CHECK(cudaMalloc(&d_sums, blocks * sizeof(float)));
    using Kernel = void (*)(const float *, int, float *);
    Kernel kernels[] = {block_sum_divergent, block_sum};
    const char *names[] = {"buggy", "fixed"};
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaMemset(d_sums, 0, blocks * sizeof(float)));
        kernels[k]<<<blocks, BLOCK>>>(d_x, n, d_sums);
        CUDA_CHECK_LAUNCH();
        auto sums = to_host(d_sums, blocks);
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            float ref = 0.0f;
            for (int i = b * BLOCK; i < n && i < (b + 1) * BLOCK; ++i) ref += x[i];
            ok &= std::fabs(sums[b] - ref) <= 1e-3f;
        }
        report("divergent", names[k], ok);
    }
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_sums));
}

static void run_launch(int n) {
    int *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_v, 0, n * sizeof(int)));
    // The bug: 2048 threads per block is over the limit of 1024, so nothing runs.
    add_index<<<blocks_for(n), 2 * 1024>>>(d_v, n);
    cudaError_t err = cudaGetLastError();
    auto v = to_host(d_v, n);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok &= v[i] == i;
    report("launch", "buggy", ok);
    std::printf("  launch    buggy  cudaGetLastError(): %s\n", cudaGetErrorString(err));
    add_index<<<blocks_for(n), BLOCK>>>(d_v, n);
    CUDA_CHECK_LAUNCH();
    v = to_host(d_v, n);
    ok = true;
    for (int i = 0; i < n; ++i) ok &= v[i] == i;
    report("launch", "fixed", ok);
    CUDA_CHECK(cudaFree(d_v));
}

struct Bug {
    const char *name;
    void (*run)(int);
};

int main(int argc, char **argv) {
    Bug bugs[] = {{"race", run_race},     {"oob", run_oob},
                  {"nosync", run_nosync}, {"uninit", run_uninit},
                  {"divergent", run_divergent}, {"launch", run_launch}};
    const char *which = argc > 1 ? argv[1] : "all";
    int n = argc > 2 ? std::atoi(argv[2]) : 1 << 20;
    if (n < 2) {
        std::fprintf(stderr, "need n >= 2\n");
        return 1;
    }
    bool found = false;
    for (auto &b : bugs) {
        if (std::strcmp(which, "all") != 0 && std::strcmp(which, b.name) != 0) continue;
        found = true;
        // One NVTX range per bug, so each shows up by name on the Nsight Systems timeline.
        nvtxRangePushA(b.name);
        b.run(n);
        nvtxRangePop();
    }
    if (!found) {
        std::fprintf(stderr, "usage: %s race|oob|nosync|uninit|divergent|launch|all [n]\n", argv[0]);
        return 1;
    }
    if (fixed_failures > 0) {
        std::printf("FAIL: %d fixed versions wrong\n", fixed_failures);
        return 1;
    }
    std::printf("OK: every fixed version matches the CPU\n");
    return 0;
}
