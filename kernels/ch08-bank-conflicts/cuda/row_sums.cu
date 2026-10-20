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

// One warp per block; the block sums 32 rows, one row per lane.
constexpr int TILE = 32;

// PAD = 0: tile[lane][j] is word lane * 32 + j, so all 32 lanes hit bank j.
// PAD = 1: the word is lane * 33 + j, so lane k hits bank (k + j) % 32.
template <int PAD>
__global__ void row_sums(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + PAD];
    int lane = threadIdx.x;
    int row0 = blockIdx.x * TILE;
    float sum = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += TILE) {
        // Lanes load 32 neighboring floats of one row: coalesced, and one tile row per store.
        for (int r = 0; r < TILE; ++r) {
            int row = row0 + r;
            tile[r][lane] = row < rows ? in[row * cols + c0 + lane] : 0.0f;
        }
        __syncthreads();
        // Each lane sums its own row, so at every j the warp reads column j of the tile.
        for (int j = 0; j < TILE; ++j) sum += tile[lane][j];
        __syncthreads();
    }
    if (row0 + lane < rows) out[row0 + lane] = sum;
}

// No padding: element (r, c) is stored in column c ^ r, which spreads every column over all 32 banks.
__global__ void row_sums_swizzled(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE];
    int lane = threadIdx.x;
    int row0 = blockIdx.x * TILE;
    float sum = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += TILE) {
        for (int r = 0; r < TILE; ++r) {
            int row = row0 + r;
            tile[r][lane ^ r] = row < rows ? in[row * cols + c0 + lane] : 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < TILE; ++j) sum += tile[lane][j ^ lane];
        __syncthreads();
    }
    if (row0 + lane < rows) out[row0 + lane] = sum;
}

using Kernel = void (*)(const float *, float *, int, int);

// Runs one kernel, checks it against the CPU sums and prints its time. Returns the mismatch count.
static int check(Kernel kernel, const char *name, const float *d_in, float *d_out,
                 const std::vector<float> &expected, int rows, int cols) {
    int blocks = (rows + TILE - 1) / TILE;
    kernel<<<blocks, TILE>>>(d_in, d_out, rows, cols);  // warm-up
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<blocks, TILE>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<float> out(rows);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, rows * sizeof(float), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < rows; ++i) errors += out[i] != expected[i];
    std::printf("%-10s %8.3f ms\n", name, ms);
    return errors;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? std::atoi(argv[1]) : 1 << 15;
    int cols = argc > 2 ? std::atoi(argv[2]) : 256;
    if (rows < 1 || cols < TILE || cols % TILE != 0) {
        std::fprintf(stderr, "usage: %s [rows] [cols, a multiple of 32]\n", argv[0]);
        return 2;
    }
    size_t n = static_cast<size_t>(rows) * cols;

    // Small integers keep every float sum exact, so any summation order matches.
    std::vector<float> in(n);
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dist(-8, 8);
    for (float &v : in) v = static_cast<float>(dist(gen));
    std::vector<float> expected(rows, 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) expected[r] += in[static_cast<size_t>(r) * cols + c];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int errors = 0;
    errors += check(row_sums<0>, "plain", d_in, d_out, expected, rows, cols);
    errors += check(row_sums<1>, "padded", d_in, d_out, expected, rows, cols);
    errors += check(row_sums_swizzled, "swizzled", d_in, d_out, expected, rows, cols);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
