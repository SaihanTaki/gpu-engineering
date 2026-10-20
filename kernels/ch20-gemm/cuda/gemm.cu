#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t st_ = (call);                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "%s:%d: cuBLAS error %d\n", __FILE__,    \
                         __LINE__, (int)st_);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

using Launch = std::function<void()>;

const int REPS = 10;

// Mean time per run in ms over REPS runs, after one untimed warm-up run.
static float time_ms(const Launch &launch) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

static dim3 tiles(int M, int N, int BM, int BN) { return dim3((N + BN - 1) / BN, (M + BM - 1) / BM); }

template <int BM, int BN, int BK, int TM, int TN>
static Launch vec(const float *A, const float *B, float *C, int M, int N, int K) {
    return [=] { gemm_vec<BM, BN, BK, TM, TN><<<tiles(M, N, BM, BN), (BM / TM) * (BN / TN)>>>(A, B, C, M, N, K); };
}

int main(int argc, char **argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 4096;
    int N = argc > 2 ? std::atoi(argv[2]) : M;
    int K = argc > 3 ? std::atoi(argv[3]) : M;
    // The kernels index with int.
    if (M < 1 || N < 1 || K < 1 || (long long)M * K > INT_MAX || (long long)K * N > INT_MAX ||
        (long long)M * N > INT_MAX) {
        std::fprintf(stderr, "need 1 <= M, N, K and every matrix below %d elements\n", INT_MAX);
        return 1;
    }
    const bool can_vec = K % 4 == 0 && N % 4 == 0;

    // Whole numbers from -4 to 4: every sum is exact in float32 in any order, so results must match exactly.
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c) hA[(size_t)r * K + c] = (float)((r * 7 + c * 13) % 9 - 4);
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c) hB[(size_t)r * N + c] = (float)((r * 5 + c * 11) % 9 - 4);

    // CPU reference on every element for small sizes, on 4096 sampled elements otherwise.
    std::vector<std::pair<int, int>> checks;
    if ((long long)M * N * K <= (1LL << 27)) {
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c) checks.push_back({r, c});
    } else {
        unsigned s = 12345;
        for (int i = 0; i < 4096; ++i) {
            s = s * 1664525u + 1013904223u;
            int r = (int)(s % (unsigned)M);
            s = s * 1664525u + 1013904223u;
            checks.push_back({r, (int)(s % (unsigned)N)});
        }
    }
    std::vector<float> ref;
    for (auto [r, c] : checks) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += (double)hA[(size_t)r * K + k] * hB[(size_t)k * N + c];
        ref.push_back((float)acc);
    }

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float one = 1.0f, zero = 0.0f;

    struct Variant {
        const char *name;
        Launch launch;
    };
    std::vector<Variant> variants = {
        {"naive", [=] { gemm_naive<<<dim3((M + 31) / 32, (N + 31) / 32), dim3(32, 32)>>>(A, B, C, M, N, K); }},
        {"coalesced", [=] { gemm_coalesced<<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 32)>>>(A, B, C, M, N, K); }},
        {"smem 32x32x32", [=] { gemm_smem<32, 32, 32><<<tiles(M, N, 32, 32), 32 * 32>>>(A, B, C, M, N, K); }},
        {"1d 64x64x8 tm8", [=] { gemm_1d<64, 64, 8, 8><<<tiles(M, N, 64, 64), 64 / 8 * 64>>>(A, B, C, M, N, K); }},
        {"2d 128x128x8 8x8", [=] { gemm_2d<128, 128, 8, 8, 8><<<tiles(M, N, 128, 128), 16 * 16>>>(A, B, C, M, N, K); }},
    };
    // Candidates for the autotuner: the vectorized kernel at several tile shapes.
    std::vector<Variant> tuned;
    if (can_vec) {
        variants.push_back({"vec 128x128x8 8x8", vec<128, 128, 8, 8, 8>(A, B, C, M, N, K)});
        tuned = {
            {"vec 64x64x16 4x4", vec<64, 64, 16, 4, 4>(A, B, C, M, N, K)},
            {"vec 64x64x8 8x8", vec<64, 64, 8, 8, 8>(A, B, C, M, N, K)},
            {"vec 128x64x16 8x4", vec<128, 64, 16, 8, 4>(A, B, C, M, N, K)},
            {"vec 64x128x16 4x8", vec<64, 128, 16, 4, 8>(A, B, C, M, N, K)},
            {"vec 128x128x16 8x8", vec<128, 128, 16, 8, 8>(A, B, C, M, N, K)},
            {"vec 128x128x8 8x4", vec<128, 128, 8, 8, 4>(A, B, C, M, N, K)},
        };
    }
    // Row-major C = A * B is column-major C^T = B^T * A^T, so cuBLAS gets B first.
    Variant cublas = {"cuBLAS", [=] {
                          CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
                      }};

    const double flops = 2.0 * M * N * K;
    std::printf("C = A * B, M = %d, N = %d, K = %d, float32, %zu checked elements\n", M, N, K, checks.size());
    if (!can_vec) std::printf("  K or N is not a multiple of 4: skipping the float4 kernels\n");
    int failures = 0;
    auto run = [&](const Variant &v) {
        CUDA_CHECK(cudaMemset(C, 0, hC.size() * sizeof(float)));
        float ms = time_ms(v.launch);
        CUDA_CHECK(cudaMemcpy(hC.data(), C, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
        int errors = 0;
        for (size_t i = 0; i < checks.size(); ++i)
            errors += hC[(size_t)checks[i].first * N + checks[i].second] != ref[i];
        failures += errors != 0;
        std::printf("  %-20s %9.3f ms  %8.1f GFLOP/s%s\n", v.name, ms, flops / (ms * 1e-3) / 1e9,
                    errors ? "  MISMATCH" : "");
        return ms;
    };
    for (auto &v : variants) run(v);
    if (!tuned.empty()) {
        std::printf("autotuning the vectorized kernel over %zu tile shapes:\n", tuned.size());
        std::vector<float> ms;
        for (auto &v : tuned) ms.push_back(run(v));
        size_t best = std::min_element(ms.begin(), ms.end()) - ms.begin();
        std::printf("  fastest: %s\n", tuned[best].name);
    }
    run(cublas);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
