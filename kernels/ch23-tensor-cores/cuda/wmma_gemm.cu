#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "wmma_gemm.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

const int REPS = 20;

// Fixed-seed generator, so every run builds the same matrices.
static float next_uniform(unsigned &s) {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
}

template <typename T>
static T *upload(const std::vector<T> &h) {
    T *d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Mean time of one launch in ms over REPS launches, after one untimed warm-up launch.
template <typename Launch>
static float time_ms(Launch launch) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPS; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

// Largest |got - want| / (sum over k of |a| |b|), the scale a dot product's rounding error is measured against.
static double max_rel_error(const std::vector<float> &got, const std::vector<double> &want,
                            const std::vector<double> &scale) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - want[i]) / scale[i]);
    return worst;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    if (n < 16 || n % 16 != 0 || n > 8192) {
        std::fprintf(stderr, "need a size that is a multiple of 16, from 16 to 8192\n");
        return 1;
    }
    int M = n, N = n, K = n;
    std::vector<half> a((size_t)M * K), b((size_t)K * N);
    // The reference uses the FP16-rounded inputs, so only the GEMM's own rounding is measured.
    std::vector<double> af(a.size()), bf(b.size());
    unsigned s = 12345;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half_rn(next_uniform(s));
        af[i] = __half2float(a[i]);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = __float2half_rn(next_uniform(s));
        bf[i] = __half2float(b[i]);
    }

    std::printf("C = A B, %d x %d x %d, FP16 inputs\n", M, N, K);
    std::vector<double> want((size_t)M * N, 0.0), scale((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            double x = af[(size_t)i * K + k];
            for (int j = 0; j < N; ++j) {
                double p = x * bf[(size_t)k * N + j];
                want[(size_t)i * N + j] += p;
                scale[(size_t)i * N + j] += std::fabs(p);
            }
        }

    half *d_a = upload(a), *d_b = upload(b), *d_c16;
    float *d_c32;
    CUDA_CHECK(cudaMalloc(&d_c32, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c16, (size_t)M * N * sizeof(half)));
    dim3 block(32 * WARPS_M * WARPS_N);
    dim3 grid(N / (WMMA_N * WARPS_N) + (N % (WMMA_N * WARPS_N) != 0), M / (WMMA_M * WARPS_M) + (M % (WMMA_M * WARPS_M) != 0));
    double flops = 2.0 * M * N * K;

    float ms32 = time_ms([&] { wmma_gemm_f32acc<<<grid, block>>>(M, N, K, d_a, d_b, d_c32); });
    std::vector<float> c32((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(c32.data(), d_c32, c32.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float ms16 = time_ms([&] { wmma_gemm_f16acc<<<grid, block>>>(M, N, K, d_a, d_b, d_c16); });
    std::vector<half> c16h((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(c16h.data(), d_c16, c16h.size() * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> c16(c16h.size());
    for (size_t i = 0; i < c16.size(); ++i) c16[i] = __half2float(c16h[i]);

    double err32 = max_rel_error(c32, want, scale);
    double err16 = max_rel_error(c16, want, scale);
    std::printf("  FP32 accumulate %9.3f ms %8.2f TFLOPS  max error %.2e\n", ms32, flops / ms32 / 1e9, err32);
    std::printf("  FP16 accumulate %9.3f ms %8.2f TFLOPS  max error %.2e\n", ms16, flops / ms16 / 1e9, err16);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c32));
    CUDA_CHECK(cudaFree(d_c16));
    // With an FP32 accumulator the error should stay below FP16's unit roundoff, 2^-11.
    if (!(err32 < std::ldexp(1.0, -11))) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
