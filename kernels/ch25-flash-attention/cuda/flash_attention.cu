#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "flash_attention_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

const int HEADS = 8;
const int REPS = 10;

// softmax(q k^T * scale) v for one head, in double, materializing each row of S.
static void reference(const float *q, const float *k, const float *v, double *o, int n, double scale, bool causal) {
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) {
        int cols = causal ? i + 1 : n;
        double mx = -INFINITY;
        for (int j = 0; j < cols; ++j) {
            double dot = 0;
            for (int c = 0; c < FA_D; ++c) dot += (double)q[i * FA_D + c] * k[j * FA_D + c];
            s[j] = dot * scale;
            mx = std::fmax(mx, s[j]);
        }
        double sum = 0;
        for (int j = 0; j < cols; ++j) sum += s[j] = std::exp(s[j] - mx);
        for (int c = 0; c < FA_D; ++c) {
            double acc = 0;
            for (int j = 0; j < cols; ++j) acc += s[j] * v[j * FA_D + c];
            o[i * FA_D + c] = acc / sum;
        }
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    int causal = argc > 2 ? std::atoi(argv[2]) : 0;
    if (n < 1 || n > (1 << 16)) {
        std::fprintf(stderr, "need 1 <= n <= 65536\n");
        return 1;
    }
    size_t count = (size_t)HEADS * n * FA_D;
    size_t bytes = count * sizeof(float);
    std::vector<float> q(count), k(count), v(count), o(count);
    std::mt19937 rng(0);
    std::normal_distribution<float> dist;
    for (size_t i = 0; i < count; ++i) q[i] = dist(rng), k[i] = dist(rng), v[i] = dist(rng);

    float *dq, *dk, *dv, *dout;
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMalloc(&dk, bytes));
    CUDA_CHECK(cudaMalloc(&dv, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), bytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)FA_D);
    dim3 grid((n + FA_BR - 1) / FA_BR, HEADS);
    auto launch = [&] { flash_attention_fwd<<<grid, FA_BR>>>(dq, dk, dv, dout, n, scale, causal); };

    // The first launch loads the kernel onto the GPU, so it is not timed.
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
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= REPS;
    CUDA_CHECK(cudaMemcpy(o.data(), dout, bytes, cudaMemcpyDeviceToHost));

    // Check the first and last head against the CPU; the others run the same code.
    std::vector<double> want((size_t)n * FA_D);
    double worst = 0;
    for (int h : {0, HEADS - 1}) {
        size_t off = (size_t)h * n * FA_D;
        reference(&q[off], &k[off], &v[off], want.data(), n, scale, causal);
        for (size_t i = 0; i < want.size(); ++i) worst = std::fmax(worst, std::fabs(o[off + i] - want[i]));
    }

    double flops = 4.0 * HEADS * n * (double)n * FA_D / (causal ? 2 : 1);
    std::printf("heads %d, n %d, d %d, causal %d, FP32, Br %d, Bc %d\n", HEADS, n, FA_D, causal, FA_BR, FA_BC);
    std::printf("  flash_attention_fwd %9.3f ms  %7.2f TFLOPS  max error %.2e\n", ms, flops / ms / 1e9, worst);
    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dout));
    if (worst > 1e-4) {
        std::printf("MISMATCH\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
