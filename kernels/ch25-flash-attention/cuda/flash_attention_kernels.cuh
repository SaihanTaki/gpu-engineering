// FlashAttention-style forward pass in FP32, without tensor cores.
// One block handles FA_BR rows of Q for one (batch, head); each thread owns one row.
// K and V stream through shared memory FA_BC rows at a time; S never reaches global memory.
#pragma once

#define FA_D 64   // head dimension, fixed so q, acc and s stay in registers
#define FA_BR 64  // rows of Q per block, one per thread
#define FA_BC 32  // rows of K and V per shared-memory tile
// -infinity without math.h, which NVRTC does not provide.
#define FA_NEG_INF __int_as_float(0xff800000)

// q, k, v, o: [heads][n][FA_D], row-major. grid = (ceil(n / FA_BR), heads), block = FA_BR threads.
extern "C" __global__ void flash_attention_fwd(const float *q, const float *k, const float *v, float *o,
                                               int n, float scale, int causal) {
    __shared__ float ks[FA_BC][FA_D];
    __shared__ float vs[FA_BC][FA_D];

    const size_t head = (size_t)blockIdx.y * n * FA_D;
    const int row = blockIdx.x * FA_BR + threadIdx.x;
    const bool active = row < n;

    float qr[FA_D], acc[FA_D];
#pragma unroll
    for (int c = 0; c < FA_D; ++c) {
        qr[c] = active ? q[head + (size_t)row * FA_D + c] * scale : 0.0f;
        acc[c] = 0.0f;
    }
    float m = FA_NEG_INF, l = 0.0f;

    // Causal: tiles that start after the block's last row are fully masked, so the loop stops there.
    const int hi = causal ? min(n, (int)(blockIdx.x + 1) * FA_BR) : n;
    for (int start = 0; start < hi; start += FA_BC) {
        __syncthreads();  // every thread is done with the previous tile
        for (int e = threadIdx.x; e < FA_BC * FA_D; e += blockDim.x) {
            const int r = e / FA_D, c = e % FA_D;
            const bool in = start + r < n;
            ks[r][c] = in ? k[head + (size_t)(start + r) * FA_D + c] : 0.0f;
            vs[r][c] = in ? v[head + (size_t)(start + r) * FA_D + c] : 0.0f;
        }
        __syncthreads();
        if (!active) continue;

        float s[FA_BC];
        float tile_max = FA_NEG_INF;
#pragma unroll
        for (int j = 0; j < FA_BC; ++j) {
            float dot = 0.0f;
#pragma unroll
            for (int c = 0; c < FA_D; ++c) dot += qr[c] * ks[j][c];
            const int col = start + j;
            if (col >= n || (causal && col > row)) dot = FA_NEG_INF;
            s[j] = dot;
            tile_max = fmaxf(tile_max, dot);
        }

        // Online softmax. The first tile always holds column 0, so m_new is finite from then on.
        const float m_new = fmaxf(m, tile_max);
        const float alpha = expf(m - m_new);
        l *= alpha;
#pragma unroll
        for (int c = 0; c < FA_D; ++c) acc[c] *= alpha;
#pragma unroll
        for (int j = 0; j < FA_BC; ++j) {
            const float p = expf(s[j] - m_new);
            l += p;
#pragma unroll
            for (int c = 0; c < FA_D; ++c) acc[c] += p * vs[j][c];
        }
        m = m_new;
    }

    if (active) {
#pragma unroll
        for (int c = 0; c < FA_D; ++c) o[head + (size_t)row * FA_D + c] = acc[c] / l;
    }
}
