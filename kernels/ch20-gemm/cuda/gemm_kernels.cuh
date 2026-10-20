#pragma once

// C = A * B for row-major float32 matrices: A is M x K, B is K x N, C is M x N.
// Device code only, so NVRTC can compile this header without nvcc.

// 1. Naive. One thread per element of C. threadIdx.x walks rows, so the 32
// lanes of a warp read 32 different rows of A: 32 sectors per load.
__global__ void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// 2. Coalesced. Same work, but threadIdx.x walks columns: the warp reads one
// word of A (a broadcast) and 32 neighbouring words of B (4 sectors).
__global__ void gemm_coalesced(const float *A, const float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// Copies the ROWS x COLS block of src at (r0, c0) into a row-major tile.
// Consecutive threads take consecutive columns, so the reads are coalesced.
// Elements past the edge become 0 and add nothing to the sums.
template <int ROWS, int COLS, int THREADS>
__device__ void load_tile(float *tile, const float *src, int r0, int c0, int nrows, int ncols) {
    for (int i = threadIdx.x; i < ROWS * COLS; i += THREADS) {
        int r = r0 + i / COLS, c = c0 + i % COLS;
        tile[i] = (r < nrows && c < ncols) ? src[r * ncols + c] : 0.0f;
    }
}

// 3. Shared memory tiling. A block of BM * BN threads computes a BM x BN tile
// of C. Each BM x BK tile of A and BK x BN tile of B is read from global
// memory once per block and then BN (or BM) times from shared memory.
template <int BM, int BN, int BK>
__global__ void gemm_smem(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int tr = threadIdx.x / BN, tc = threadIdx.x % BN;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile<BM, BK, BM * BN>(As, A, row0, k0, M, K);
        load_tile<BK, BN, BM * BN>(Bs, B, k0, col0, K, N);
        __syncthreads();
        for (int k = 0; k < BK; ++k)
            acc += As[tr * BK + k] * Bs[k * BN + tc];
        // The next iteration overwrites the tiles.
        __syncthreads();
    }
    if (row0 + tr < M && col0 + tc < N)
        C[(row0 + tr) * N + col0 + tc] = acc;
}

// 4. 1D register blocking. Each thread computes TM elements of one column of
// the tile. One value of B, held in a register, feeds TM multiply-adds.
template <int BM, int BN, int BK, int TM>
__global__ void gemm_1d(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int THREADS = BM / TM * BN;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int tr = threadIdx.x / BN, tc = threadIdx.x % BN;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile<BM, BK, THREADS>(As, A, row0, k0, M, K);
        load_tile<BK, BN, THREADS>(Bs, B, k0, col0, K, N);
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            float b = Bs[k * BN + tc];
            for (int i = 0; i < TM; ++i)
                acc[i] += As[(tr * TM + i) * BK + k] * b;
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        int row = row0 + tr * TM + i, col = col0 + tc;
        if (row < M && col < N)
            C[row * N + col] = acc[i];
    }
}

// 5. 2D register blocking. Each thread computes a TM x TN block of C from TM
// values of A and TN values of B in registers: TM * TN multiply-adds for
// TM + TN shared memory loads.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_2d(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile<BM, BK, THREADS>(As, A, row0, k0, M, K);
        load_tile<BK, BN, THREADS>(Bs, B, k0, col0, K, N);
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) a[i] = As[(tr * TM + i) * BK + k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k * BN + tc * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int row = row0 + tr * TM + i, col = col0 + tc * TN + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
}

// 6. Vectorized. The 2D kernel with 16-byte float4 loads and stores. A's tile
// is stored transposed, so a thread's TM values of A are contiguous in shared
// memory too. Needs K and N to be multiples of 4 and TM, TN, BK, BN as well,
// so every float4 is either fully inside the matrix or fully outside.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_vec(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0 && BN % 4 == 0, "float4 needs multiples of 4");
    __shared__ __align__(16) float AsT[BK * BM];
    __shared__ __align__(16) float Bs[BK * BN];
    const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK / 4; i += THREADS) {
            int r = i / (BK / 4), c = i % (BK / 4) * 4;
            float4 v = (row0 + r < M && k0 + c < K)
                ? *reinterpret_cast<const float4 *>(&A[(row0 + r) * K + k0 + c]) : zero;
            AsT[(c + 0) * BM + r] = v.x;
            AsT[(c + 1) * BM + r] = v.y;
            AsT[(c + 2) * BM + r] = v.z;
            AsT[(c + 3) * BM + r] = v.w;
        }
        for (int i = threadIdx.x; i < BK * BN / 4; i += THREADS) {
            int r = i / (BN / 4), c = i % (BN / 4) * 4;
            *reinterpret_cast<float4 *>(&Bs[r * BN + c]) = (k0 + r < K && col0 + c < N)
                ? *reinterpret_cast<const float4 *>(&B[(k0 + r) * N + col0 + c]) : zero;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            // Copied out by component: a and b must stay in registers.
            for (int i = 0; i < TM; i += 4) {
                float4 v = *reinterpret_cast<const float4 *>(&AsT[k * BM + tr * TM + i]);
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            for (int j = 0; j < TN; j += 4) {
                float4 v = *reinterpret_cast<const float4 *>(&Bs[k * BN + tc * TN + j]);
                b[j] = v.x; b[j + 1] = v.y; b[j + 2] = v.z; b[j + 3] = v.w;
            }
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        int row = row0 + tr * TM + i;
        for (int j = 0; j < TN; j += 4) {
            int col = col0 + tc * TN + j;
            if (row < M && col < N)
                *reinterpret_cast<float4 *>(&C[row * N + col]) = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
        }
    }
}
