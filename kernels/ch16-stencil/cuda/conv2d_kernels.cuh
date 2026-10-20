// 2D convolution of an h x w row-major float image with a (2R+1) x (2R+1)
// filter f:
//
//     out[y][x] = sum over dy, dx in [-R, R] of f[dy + R][dx + R] * in[y + dy][x + dx]
//
// Pixels outside the image count as zero (zero padding), so out has the same
// size as in. The filter is not flipped, the convention of deep learning
// libraries; for a symmetric filter the two conventions agree.
//
// The header holds only device code, so NVRTC can compile it without nvcc.
#pragma once

// An enum, not constexpr variables: NVRTC would emit those as device globals.
enum { TILE = 16, MAX_R = 3 };

// Every thread of a warp reads the same weight at the same time, which the
// constant cache serves as one broadcast. The host fills the first
// (2R+1) * (2R+1) entries, row by row.
__constant__ float c_filter[(2 * MAX_R + 1) * (2 * MAX_R + 1)];

// One thread per output pixel; each one loads its whole window from global memory.
template <int R>
__global__ void conv2d_naive(const float *in, float *out, int h, int w) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < h && xx >= 0 && xx < w)
                sum += c_filter[(dy + R) * (2 * R + 1) + dx + R] * in[yy * w + xx];
        }
    out[y * w + x] = sum;
}

// A block of TILE x TILE threads computes a TILE x TILE output tile. It first
// copies the input tile plus a halo of R pixels on every side into shared
// memory, so each input pixel is loaded from global memory once per block.
template <int R>
__global__ void conv2d_tiled(const float *in, float *out, int h, int w) {
    constexpr int IN = TILE + 2 * R;
    __shared__ float tile[IN][IN];
    // Image coordinates of tile[0][0], the top-left corner of the halo.
    int x0 = blockIdx.x * TILE - R;
    int y0 = blockIdx.y * TILE - R;
    // IN * IN slots and only TILE * TILE threads: each thread fills every
    // (TILE * TILE)-th slot, row by row.
    for (int i = threadIdx.y * TILE + threadIdx.x; i < IN * IN; i += TILE * TILE) {
        int yy = y0 + i / IN, xx = x0 + i % IN;
        tile[i / IN][i % IN] = (yy >= 0 && yy < h && xx >= 0 && xx < w) ? in[yy * w + xx] : 0.0f;
    }
    __syncthreads();

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    // Past the only barrier, so threads outside the image may stop here.
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int dy = 0; dy <= 2 * R; ++dy)
        for (int dx = 0; dx <= 2 * R; ++dx)
            sum += c_filter[dy * (2 * R + 1) + dx] * tile[threadIdx.y + dy][threadIdx.x + dx];
    out[y * w + x] = sum;
}
