// generate.py compiles the kernels in this file with NVRTC, which has no host
// compiler, so the host code is hidden from it.
#ifndef __CUDACC_RTC__
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#endif

// An enum, not constexpr variables: NVRTC would emit those as device globals.
enum { DEGREE = 8, BINS = 16, KEYS_PER_THREAD = 64, LIVE = 64 };

// Every thread of a warp reads the same coefficient at the same time,
// which the constant cache serves as one broadcast.
__constant__ float c_coef[DEGREE + 1];

// y = c[0] + c[1] x + ... + c[DEGREE] x^DEGREE, Horner's rule.
__global__ void poly_constant(int n, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], acc = c_coef[DEGREE];
    for (int k = DEGREE - 1; k >= 0; --k) acc = acc * v + c_coef[k];
    y[i] = acc;
}

// The same polynomial with the coefficients in global memory. const and
// __restrict__ tell the compiler nothing writes x or coef during the kernel,
// so it may load them through the read-only path (ld.global.nc).
__global__ void poly_readonly(int n, const float *__restrict__ x, const float *__restrict__ coef,
                              float *__restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], acc = __ldg(&coef[DEGREE]);
    for (int k = DEGREE - 1; k >= 0; --k) acc = acc * v + __ldg(&coef[k]);
    y[i] = acc;
}

// Each thread counts the keys of its own chunk. bins[] is indexed by data,
// which registers cannot do, so it lives in local memory.
__global__ void bin_counts(int threads, const int *keys, int *counts) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;
    int bins[BINS] = {};
    for (int k = 0; k < KEYS_PER_THREAD; ++k) bins[keys[t * KEYS_PER_THREAD + k] % BINS]++;
    for (int b = 0; b < BINS; ++b) counts[t * BINS + b] = bins[b];
}

// out[i] = sum of v[k] * v[LIVE-1-k] over LIVE values loaded by thread i.
// Half of the values must stay live until the other half arrives.
__device__ __forceinline__ float mirror_sum(int n, const float *x, int i) {
    float v[LIVE];
#pragma unroll
    for (int k = 0; k < LIVE; ++k) v[k] = x[k * n + i];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < LIVE; ++k) s += v[k] * v[LIVE - 1 - k];
    return s;
}

__global__ void mirror(int n, const float *x, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mirror_sum(n, x, i);
}

// The same code, capped at 32 registers per thread: the rest spill.
__global__ void __maxnreg__(32) mirror_capped(int n, const float *x, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mirror_sum(n, x, i);
}

#ifndef __CUDACC_RTC__
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

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

static bool close(const std::vector<float> &got, const std::vector<float> &want, const char *name) {
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-4f * (1.0f + std::fabs(want[i]))) {
            std::fprintf(stderr, "%s: mismatch at %zu: %f vs %f\n", name, i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

static void report(const char *name, const void *kernel) {
    cudaFuncAttributes a;
    CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    std::printf("kernel %s regs=%d local_bytes=%zu\n", name, a.numRegs, a.localSizeBytes);
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    const int block = 256, grid = (n + block - 1) / block;
    bool ok = true;

    std::vector<float> x(n), coef(DEGREE + 1);
    for (int i = 0; i < n; ++i) x[i] = std::sin(0.001f * i);
    for (int k = 0; k <= DEGREE; ++k) coef[k] = 1.0f / (k + 1);
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) {
        float acc = coef[DEGREE];
        for (int k = DEGREE - 1; k >= 0; --k) acc = acc * x[i] + coef[k];
        want[i] = acc;
    }

    float *d_x = to_device(x), *d_coef = to_device(coef), *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_coef, coef.data(), coef.size() * sizeof(float)));
    poly_constant<<<grid, block>>>(n, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    ok &= close(to_host(d_y, n), want, "poly_constant");
    poly_readonly<<<grid, block>>>(n, d_x, d_coef, d_y);
    CUDA_CHECK(cudaGetLastError());
    ok &= close(to_host(d_y, n), want, "poly_readonly");

    const int threads = 4096;
    std::vector<int> keys(threads * KEYS_PER_THREAD);
    for (size_t k = 0; k < keys.size(); ++k) keys[k] = (int)(((unsigned)k * 2654435761u) >> 8);
    int *d_keys = to_device(keys), *d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, threads * BINS * sizeof(int)));
    bin_counts<<<threads / block, block>>>(threads, d_keys, d_counts);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> counts = to_host(d_counts, threads * BINS);
    for (int t = 0; t < threads && ok; ++t) {
        int c[BINS] = {};
        for (int k = 0; k < KEYS_PER_THREAD; ++k) c[keys[t * KEYS_PER_THREAD + k] % BINS]++;
        for (int b = 0; b < BINS; ++b) ok &= counts[t * BINS + b] == c[b];
    }
    if (!ok) std::fprintf(stderr, "bin_counts: mismatch\n");

    const int m = 1 << 14;
    std::vector<float> xs((size_t)LIVE * m), mw(m);
    for (size_t k = 0; k < xs.size(); ++k) xs[k] = std::cos(0.01f * k);
    for (int i = 0; i < m; ++i) {
        float s = 0.0f;
        for (int k = 0; k < LIVE; ++k) s += xs[(size_t)k * m + i] * xs[(size_t)(LIVE - 1 - k) * m + i];
        mw[i] = s;
    }
    float *d_xs = to_device(xs), *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, m * sizeof(float)));
    mirror<<<m / block, block>>>(m, d_xs, d_out);
    CUDA_CHECK(cudaGetLastError());
    ok &= close(to_host(d_out, m), mw, "mirror");
    mirror_capped<<<m / block, block>>>(m, d_xs, d_out);
    CUDA_CHECK(cudaGetLastError());
    ok &= close(to_host(d_out, m), mw, "mirror_capped");

    report("poly_constant", (const void *)poly_constant);
    report("poly_readonly", (const void *)poly_readonly);
    report("bin_counts", (const void *)bin_counts);
    report("mirror", (const void *)mirror);
    report("mirror_capped", (const void *)mirror_capped);

    cudaFree(d_x); cudaFree(d_coef); cudaFree(d_y); cudaFree(d_keys); cudaFree(d_counts);
    cudaFree(d_xs); cudaFree(d_out);
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
#endif
