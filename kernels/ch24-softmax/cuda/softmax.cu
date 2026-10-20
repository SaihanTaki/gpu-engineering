#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "softmax_kernels.cuh"

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
const int WARPS_PER_BLOCK = 4;
const int REPS = 20;
const float EPS = 1e-5f;

// CPU references in double precision.
static void softmax_ref(const std::vector<float> &x, std::vector<double> &y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *row = &x[(size_t)r * cols];
        double m = -INFINITY, d = 0.0;
        for (int i = 0; i < cols; ++i) m = std::fmax(m, row[i]);
        for (int i = 0; i < cols; ++i) d += std::exp(row[i] - m);
        for (int i = 0; i < cols; ++i) y[(size_t)r * cols + i] = std::exp(row[i] - m) / d;
    }
}

static void layernorm_ref(const std::vector<float> &x, const std::vector<float> &g, const std::vector<float> &b,
                          std::vector<double> &y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *row = &x[(size_t)r * cols];
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < cols; ++i) mean += row[i];
        mean /= cols;
        for (int i = 0; i < cols; ++i) var += (row[i] - mean) * (row[i] - mean);
        var /= cols;
        double rstd = 1.0 / std::sqrt(var + EPS);
        for (int i = 0; i < cols; ++i) y[(size_t)r * cols + i] = (row[i] - mean) * rstd * g[i] + b[i];
    }
}

static double max_error(const std::vector<float> &got, const std::vector<double> &want) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double e = std::fabs(got[i] - want[i]);
        if (std::isnan(e)) return e;
        if (e > worst) worst = e;
    }
    return worst;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
    if (rows < 2 || cols < 1) {
        std::fprintf(stderr, "usage: softmax [rows >= 2] [cols >= 1]\n");
        return 1;
    }
    size_t n = (size_t)rows * cols;

    std::mt19937 gen(0);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    std::vector<float> h_x(n), h_g(cols), h_b(cols);
    for (auto &v : h_x) v = normal(gen);
    // Row 0: one value whose exp overflows float32. Row 1: a large mean and a small spread.
    h_x[cols / 2] = 1000.0f;
    for (int i = 0; i < cols; ++i) h_x[(size_t)cols + i] = 1000.0f + normal(gen) / 3.0f;
    for (int i = 0; i < cols; ++i) {
        h_g[i] = 1.0f + 0.1f * normal(gen);
        h_b[i] = 0.1f * normal(gen);
    }

    std::vector<double> want_sm(n), want_ln(n);
    softmax_ref(h_x, want_sm, rows, cols);
    layernorm_ref(h_x, h_g, h_b, want_ln, rows, cols);

    float *d_x, *d_y, *d_g, *d_b, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int warp_blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    struct Variant {
        const char *name;
        const std::vector<double> *want;
        double tol;
    } variants[] = {
        {"softmax_safe", &want_sm, 1e-6},
        {"softmax_online", &want_sm, 1e-6},
        {"softmax_online_warp", &want_sm, 1e-6},
        {"layernorm_welford", &want_ln, 1e-3},
    };
    auto launch = [&](int v) {
        switch (v) {
            case 0: softmax_safe<<<rows, THREADS>>>(d_x, d_y, cols); break;
            case 1: softmax_online<<<rows, THREADS>>>(d_x, d_y, cols); break;
            case 2: softmax_online_warp<<<warp_blocks, 32 * WARPS_PER_BLOCK>>>(d_x, d_y, rows, cols); break;
            case 3: layernorm_welford<<<rows, THREADS>>>(d_x, d_g, d_b, d_y, d_mean, d_rstd, cols, EPS); break;
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Effective bandwidth counts the row read once and written once, the least any version can move.
    std::printf("rows = %d, cols = %d, %d threads per block\n", rows, cols, THREADS);
    std::vector<float> got(n);
    int failures = 0;
    for (int v = 0; v < 4; ++v) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        launch(v);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
        double err = max_error(got, *variants[v].want);
        bool ok = err <= variants[v].tol;
        failures += !ok;

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < REPS; ++r) launch(v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= REPS;
        std::printf("  %-20s %8.3f ms  %8.1f GB/s  max error %.2e%s\n", variants[v].name, ms,
                    2.0 * n * sizeof(float) / (ms * 1e-3) / 1e9, err, ok ? "" : "  MISMATCH");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
