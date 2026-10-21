#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "spmv_kernels.cuh"

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
const int BLOCK = 256;

struct Csr {
    int rows, cols;
    std::vector<int> row_ptr, col;
    std::vector<float> val;
};

// Fixed-seed generator, so every run builds the same matrix.
static unsigned next_rand(unsigned &s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

// Rows of 1 to 16 nonzeros. With `skewed`, every 1024th row has `cols / 8` instead.
static Csr make_matrix(int rows, int cols, bool skewed) {
    Csr a{rows, cols, {0}, {}, {}};
    unsigned s = 12345;
    for (int r = 0; r < rows; ++r) {
        int len = (skewed && r % 1024 == 0) ? cols / 8 : 1 + next_rand(s) % 16;
        for (int j = 0; j < len; ++j) {
            // Columns may repeat within a row; CSR allows it and the sum is still well defined.
            a.col.push_back(next_rand(s) % cols);
            a.val.push_back((float)(next_rand(s) % 1000) / 1000.0f);
        }
        a.row_ptr.push_back((int)a.col.size());
    }
    return a;
}

static int ell_width(const Csr &a) {
    int width = 0;
    for (int r = 0; r < a.rows; ++r) width = std::max(width, a.row_ptr[r + 1] - a.row_ptr[r]);
    return width;
}

// Column-major ELL: slot k of row r at k * rows + r, padded with column 0 and value 0.
static void to_ell(const Csr &a, int width, std::vector<int> &col, std::vector<float> &val) {
    col.assign((size_t)width * a.rows, 0);
    val.assign((size_t)width * a.rows, 0.0f);
    for (int r = 0; r < a.rows; ++r) {
        for (int i = a.row_ptr[r]; i < a.row_ptr[r + 1]; ++i) {
            size_t slot = (size_t)(i - a.row_ptr[r]) * a.rows + r;
            col[slot] = a.col[i];
            val[slot] = a.val[i];
        }
    }
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

static int check(const std::vector<float> &got, const std::vector<double> &want) {
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-3 * (1.0 + std::fabs(want[i]))) ++bad;
    }
    return bad;
}

static int run(const char *name, const Csr &a) {
    std::vector<float> x(a.cols);
    for (int c = 0; c < a.cols; ++c) x[c] = (float)(c % 7) - 3.0f;
    std::vector<double> want(a.rows, 0.0);
    for (int r = 0; r < a.rows; ++r)
        for (int i = a.row_ptr[r]; i < a.row_ptr[r + 1]; ++i) want[r] += (double)a.val[i] * x[a.col[i]];

    int nnz = a.row_ptr[a.rows];
    std::printf("%s: %d x %d, %d nonzeros\n", name, a.rows, a.cols, nnz);
    int *d_row_ptr = upload(a.row_ptr), *d_col = upload(a.col);
    float *d_val = upload(a.val), *d_x = upload(x), *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, a.rows * sizeof(float)));
    std::vector<float> y(a.rows);
    int failures = 0;

    auto report = [&](const char *kernel, float ms) {
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, a.rows * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = check(y, want);
        failures += bad != 0;
        std::printf("  %-12s %9.3f ms%s\n", kernel, ms, bad ? "  MISMATCH" : "");
    };

    int rows = a.rows;
    int scalar_grid = (rows + BLOCK - 1) / BLOCK;
    report("csr scalar", time_ms([&] {
        spmv_csr_scalar<<<scalar_grid, BLOCK>>>(rows, d_row_ptr, d_col, d_val, d_x, d_y);
    }));
    int vector_grid = (int)(((long long)rows * WARP + BLOCK - 1) / BLOCK);
    report("csr vector", time_ms([&] {
        spmv_csr_vector<<<vector_grid, BLOCK>>>(rows, d_row_ptr, d_col, d_val, d_x, d_y);
    }));

    int width = ell_width(a);
    double slots = (double)width * rows;
    std::printf("  ELL width %d: %.0f slots, %.1f per nonzero\n", width, slots, slots / nnz);
    // Padding a few very long rows can need far more memory than the matrix itself.
    if (slots <= 4.0 * nnz) {
        std::vector<int> ell_col;
        std::vector<float> ell_val;
        to_ell(a, width, ell_col, ell_val);
        int *d_ell_col = upload(ell_col);
        float *d_ell_val = upload(ell_val);
        report("ell", time_ms([&] {
            spmv_ell<<<scalar_grid, BLOCK>>>(rows, width, d_ell_col, d_ell_val, d_x, d_y);
        }));
        CUDA_CHECK(cudaFree(d_ell_col));
        CUDA_CHECK(cudaFree(d_ell_val));
    } else {
        std::printf("  %-12s skipped: more than 4 slots per nonzero\n", "ell");
    }

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return failures;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    if (rows < 1 || rows > (1 << 24)) {
        std::fprintf(stderr, "need 1 <= rows <= %d\n", 1 << 24);
        return 1;
    }
    int failures = run("uniform", make_matrix(rows, rows, false));
    failures += run("skewed", make_matrix(rows, rows, true));
    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
