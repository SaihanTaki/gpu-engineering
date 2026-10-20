// Three y = A x kernels for a sparse rows x cols float matrix A.
// CSR: row r's nonzeros are val[row_ptr[r] .. row_ptr[r + 1] - 1], in columns col[...].
// ELL: every row padded to `width` slots, stored column-major: slot k of row r is at k * rows + r.
#pragma once

constexpr int WARP = 32;

// One thread per row.
__global__ void spmv_csr_scalar(int rows, const int *row_ptr, const int *col, const float *val,
                                const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        // Each thread loops as long as its own row; the warp waits for the longest.
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
            sum += val[i] * x[col[i]];
        }
        y[row] = sum;
    }
}

// One warp per row. blockDim.x must be a multiple of 32.
__global__ void spmv_csr_vector(int rows, const int *row_ptr, const int *col, const float *val,
                                const float *x, float *y) {
    int lane = threadIdx.x % WARP;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    // row is the same for all 32 lanes, so whole warps leave and every shuffle sees all lanes.
    if (row >= rows) return;
    float sum = 0.0f;
    // Neighbouring lanes read neighbouring entries of col and val.
    for (int i = row_ptr[row] + lane; i < row_ptr[row + 1]; i += WARP) {
        sum += val[i] * x[col[i]];
    }
    for (int offset = WARP / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        y[row] = sum;
    }
}

// One thread per row. Padded slots hold column 0 and value 0.
__global__ void spmv_ell(int rows, int width, const int *col, const float *val, const float *x,
                         float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        // Column-major: at each k, neighbouring threads read neighbouring addresses.
        for (int k = 0; k < width; ++k) {
            int i = k * rows + row;
            sum += val[i] * x[col[i]];
        }
        y[row] = sum;
    }
}
