// Three transposes of a rows x cols row-major matrix into a cols x rows one.
// Each block of TILE x BLOCK_ROWS threads handles one TILE x TILE tile, so
// every thread moves TILE / BLOCK_ROWS = 4 elements.
#pragma once

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            // The read is coalesced; neighbouring lanes write addresses `rows` floats apart.
            out[x * rows + (y + j)] = in[(y + j) * cols + x];
        }
    }
}

// PAD = 0 is the plain tiled kernel, PAD = 1 the conflict-free one.
template <int PAD>
__global__ void transpose_shared(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }

    // Each thread now reads elements that other warps wrote.
    __syncthreads();

    // Swap the block indices: this block writes the mirrored tile of out.
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            // A warp reads a column of the tile: 32 different rows of shared memory.
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
