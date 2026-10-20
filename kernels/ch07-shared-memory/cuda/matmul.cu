#include <cmath>
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

// Blocks are TILE x TILE threads, one per element of a TILE x TILE tile of C.
constexpr int TILE = 16;

// C = A * B for n x n row-major matrices. Every thread reads a row of A and a
// column of B from global memory: 2n loads per output element.
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

// The same product, one tile of A and one of B at a time through shared
// memory: 2n / TILE global loads per output element.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    // No early return for threads past the edge: every thread must reach both barriers.
    for (int p = 0; p < (n + TILE - 1) / TILE; ++p) {
        int a_col = p * TILE + tx;
        int b_row = p * TILE + ty;
        // Zeros past the edge add nothing to the sum, so the inner loop needs no bounds check.
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        // The whole tile must be written before any thread reads it.
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        // Every thread must be done with this tile before the next phase overwrites it.
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

static void reference(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += double(A[i * n + k]) * B[k * n + j];
            C[i * n + j] = float(sum);
        }
}

using Kernel = void (*)(const float *, const float *, float *, int);

// Runs one kernel, checks it against the CPU and returns the mismatch count.
static int run(Kernel kernel, const char *name, const float *dA, const float *dB, float *dC,
               const std::vector<float> &expected, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemset(dC, 0, size_t(n) * n * sizeof(float)));
    // Warm-up launch, so the timed one does not include module loading.
    kernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<float> C(size_t(n) * n);
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (size_t i = 0; i < C.size(); ++i) {
        // float sums in a different order than the double reference; allow for rounding.
        float tol = 1e-4f * n + 1e-5f * std::fabs(expected[i]);
        errors += std::fabs(C[i] - expected[i]) > tol;
    }
    std::printf("%-6s n=%d: %.3f ms, %d mismatches\n", name, n, ms, errors);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return errors;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    size_t count = size_t(n) * n;
    std::vector<float> A(count), B(count), expected(count);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (float &v : A) v = uniform(gen);
    for (float &v : B) v = uniform(gen);
    reference(A, B, expected, n);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    int errors = run(matmul_naive, "naive", dA, dB, dC, expected, n);
    errors += run(matmul_tiled, "tiled", dA, dB, dC, expected, n);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
