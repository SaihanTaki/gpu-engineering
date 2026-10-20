#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "scan_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

using ScanKernel = void (*)(const int *, int *, int *, int);

struct Algo {
    const char *name;
    ScanKernel inclusive;
    ScanKernel exclusive;
    int per_block;
};

const Algo ALGOS[] = {
    {"hillis-steele", scan_hillis_steele<true>, scan_hillis_steele<false>, BLOCK},
    {"blelloch", scan_blelloch<true>, scan_blelloch<false>, ITEMS},
};

// Scan-then-add: scan every tile, scan the tile totals, add each tile's offset.
// scratch holds 2 ints per tile at this level and below, so nothing is allocated while timing.
static void scan_device(const Algo &algo, bool inclusive, const int *d_in, int *d_out, int n, int *scratch) {
    int tiles = (n + algo.per_block - 1) / algo.per_block;
    ScanKernel k = inclusive ? algo.inclusive : algo.exclusive;
    if (tiles == 1) {
        k<<<1, BLOCK>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int *sums = scratch;
    int *offsets = scratch + tiles;
    k<<<tiles, BLOCK>>>(d_in, d_out, sums, n);
    CUDA_CHECK(cudaGetLastError());
    // The same scan, exclusive, on the tile totals; recursion handles more than one tile of them.
    scan_device(algo, false, sums, offsets, tiles, scratch + 2 * tiles);
    add_offsets<<<(n + 255) / 256, 256>>>(d_out, offsets, n, algo.per_block);
    CUDA_CHECK(cudaGetLastError());
}

const int REPS = 20;

int main(int argc, char **argv) {
    long long n_arg = argc > 1 ? std::atoll(argv[1]) : (1 << 24);
    // Inputs are 0..9, so every prefix sum must fit in an int.
    if (n_arg < 1 || n_arg > INT_MAX / 9) {
        std::fprintf(stderr, "need 1 <= n <= %d\n", INT_MAX / 9);
        return 1;
    }
    int n = (int)n_arg;
    size_t bytes = (size_t)n * sizeof(int);

    std::vector<int> h_in(n), h_out(n), incl(n), excl(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = (int)((i * 2654435761u) >> 28) % 10;
    }
    int running = 0;
    for (int i = 0; i < n; ++i) {
        excl[i] = running;
        running += h_in[i];
        incl[i] = running;
    }

    int *d_in, *d_out, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    // 2 * (n / 256 + n / 256^2 + ...) rounded up per level stays below n / 64 + 64.
    CUDA_CHECK(cudaMalloc(&d_scratch, (n / 64 + 64) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Effective bandwidth counts each element read once and written once.
    std::printf("n = %d ints, %d threads per block\n", n, BLOCK);
    int failures = 0;
    for (const Algo &algo : ALGOS) {
        for (bool inclusive : {true, false}) {
            CUDA_CHECK(cudaMemset(d_out, 0, bytes));
            // The first run loads the kernels and is checked, not timed.
            scan_device(algo, inclusive, d_in, d_out, n, d_scratch);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
            const std::vector<int> &want = inclusive ? incl : excl;
            int errors = 0;
            for (int i = 0; i < n; ++i) {
                errors += h_out[i] != want[i];
            }
            failures += errors != 0;

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < REPS; ++r) {
                scan_device(algo, inclusive, d_in, d_out, n, d_scratch);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= REPS;
            std::printf("  %-14s %-9s %9.3f ms  %8.1f GB/s%s\n", algo.name, inclusive ? "inclusive" : "exclusive",
                        ms, 2.0 * bytes / (ms * 1e-3) / 1e9, errors ? "  MISMATCH" : "");
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
