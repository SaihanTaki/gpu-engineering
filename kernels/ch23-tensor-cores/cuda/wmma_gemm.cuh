// C = A B on tensor cores with the WMMA API. A is M x K, B is K x N, C is M x N,
// all row-major; M, N and K must be multiples of 16. Needs sm_70 or later.
// Each warp computes one 16 x 16 tile of C; a block of 4 warps covers 32 x 32.
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Warps per block along M and N.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;

template <typename Acc>
__device__ void wmma_tile(int M, int N, int K, const half *A, const half *B, Acc *C) {
    int warp = threadIdx.x / 32;
    int tile_m = (blockIdx.y * WARPS_M + warp / WARPS_N) * WMMA_M;
    int tile_n = (blockIdx.x * WARPS_N + warp % WARPS_N) * WMMA_N;
    // The condition is the same for all 32 lanes of a warp, as WMMA requires.
    if (tile_m >= M || tile_n >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Acc> acc;
    wmma::fill_fragment(acc, Acc(0.0f));

    for (int k = 0; k < K; k += WMMA_K) {
        // The last argument is the leading dimension: elements between consecutive rows.
        wmma::load_matrix_sync(a, A + (size_t)tile_m * K + k, K);
        wmma::load_matrix_sync(b, B + (size_t)k * N + tile_n, N);
        // acc = a * b + acc, by the whole warp at once.
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + (size_t)tile_m * N + tile_n, acc, N, wmma::mem_row_major);
}

// FP16 inputs, FP32 accumulator: the setting mixed-precision training uses.
__global__ void wmma_gemm_f32acc(int M, int N, int K, const half *A, const half *B, float *C) {
    wmma_tile<float>(M, N, K, A, B, C);
}

// FP16 inputs, FP16 accumulator: the rounding error grows with K.
__global__ void wmma_gemm_f16acc(int M, int N, int K, const half *A, const half *B, half *C) {
    wmma_tile<half>(M, N, K, A, B, C);
}
