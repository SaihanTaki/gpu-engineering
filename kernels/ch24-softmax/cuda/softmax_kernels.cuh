// Row softmax and LayerNorm over a row-major [rows, cols] float32 matrix.
// Device code only, so NVRTC can compile it without nvcc.
#pragma once

#define FULL_MASK 0xffffffffu
#define NEG_INF __int_as_float(0xff800000)

__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

// Chapter 13's block reduction: shuffles inside each warp, one value per warp
// through shared memory, then every warp reduces those so all threads get the
// result. blockDim.x must be a multiple of 32; scratch holds 32 floats.
__device__ float block_max(float v, float *scratch) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_max(v);
    __syncthreads();  // scratch may still be read by an earlier call
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    return warp_max(lane < blockDim.x / 32 ? scratch[lane] : NEG_INF);
}

__device__ float block_sum(float v, float *scratch) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    return warp_sum(lane < blockDim.x / 32 ? scratch[lane] : 0.0f);
}

// Safe softmax, one block per row: a max pass, a sum pass and a normalize pass,
// so each element is read three times and written once.
extern "C" __global__ void softmax_safe(const float *x, float *y, int cols) {
    __shared__ float scratch[32];
    const float *row = x + (size_t)blockIdx.x * cols;
    float *out = y + (size_t)blockIdx.x * cols;

    float m = NEG_INF;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, row[i]);
    m = block_max(m, scratch);

    float d = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) d += expf(row[i] - m);
    d = block_sum(d, scratch);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = expf(row[i] - m) / d;
}

// A running max m and the sum d of exp(x - m) over the values seen so far.
struct MD {
    float m, d;
};

// Milakov and Gimelshein's merge: rescale each sum to the larger max, then add.
__device__ MD md_merge(MD a, MD b) {
    float m = fmaxf(a.m, b.m);
    if (m == NEG_INF) return {m, 0.0f};  // both empty: avoids exp(-inf - -inf) = NaN
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

__device__ MD warp_md(MD t) {
    for (int o = 16; o > 0; o /= 2) {
        MD other = {__shfl_xor_sync(FULL_MASK, t.m, o), __shfl_xor_sync(FULL_MASK, t.d, o)};
        t = md_merge(t, other);
    }
    return t;
}

// Online softmax, one block per row: one pass finds m and d together, a second
// pass normalizes, so each element is read twice and written once.
extern "C" __global__ void softmax_online(const float *x, float *y, int cols) {
    __shared__ MD scratch[32];
    const float *row = x + (size_t)blockIdx.x * cols;
    float *out = y + (size_t)blockIdx.x * cols;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

    MD t = {NEG_INF, 0.0f};
    for (int i = threadIdx.x; i < cols; i += blockDim.x) t = md_merge(t, {row[i], 1.0f});
    t = warp_md(t);
    if (lane == 0) scratch[warp] = t;
    __syncthreads();
    t = lane < blockDim.x / 32 ? scratch[lane] : MD{NEG_INF, 0.0f};
    t = warp_md(t);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = expf(row[i] - t.m) / t.d;
}

// Online softmax, one warp per row: for short rows a whole block would leave
// most threads idle, and a warp needs no shared memory or __syncthreads().
extern "C" __global__ void softmax_online_warp(const float *x, float *y, int rows, int cols) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (r >= rows) return;
    const float *row = x + (size_t)r * cols;
    float *out = y + (size_t)r * cols;
    int lane = threadIdx.x % 32;

    MD t = {NEG_INF, 0.0f};
    for (int i = lane; i < cols; i += 32) t = md_merge(t, {row[i], 1.0f});
    t = warp_md(t);
    for (int i = lane; i < cols; i += 32) out[i] = expf(row[i] - t.m) / t.d;
}

// Count, mean and M2 (sum of squared deviations from the mean) of a set of values.
struct Welford {
    float n, mean, m2;
};

// Chan, Golub and LeVeque's pairwise merge of two partial results.
__device__ Welford welford_merge(Welford a, Welford b) {
    float n = a.n + b.n;
    if (n == 0.0f) return a;
    float delta = b.mean - a.mean;
    float wb = b.n / n;
    return {n, a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.n * wb};
}

__device__ Welford warp_welford(Welford t) {
    for (int o = 16; o > 0; o /= 2) {
        Welford other = {__shfl_xor_sync(FULL_MASK, t.n, o), __shfl_xor_sync(FULL_MASK, t.mean, o),
                         __shfl_xor_sync(FULL_MASK, t.m2, o)};
        t = welford_merge(t, other);
    }
    return t;
}

// LayerNorm forward, one block per row, in one kernel: Welford statistics in
// one pass, then normalize, scale by gamma and shift by beta in a second pass.
// mean_out and rstd_out keep one value per row, which the backward pass reuses.
extern "C" __global__ void layernorm_welford(const float *x, const float *gamma, const float *beta, float *y,
                                             float *mean_out, float *rstd_out, int cols, float eps) {
    __shared__ Welford scratch[32];
    const float *row = x + (size_t)blockIdx.x * cols;
    float *out = y + (size_t)blockIdx.x * cols;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

    Welford t = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row[i];
        t.n += 1.0f;
        float delta = v - t.mean;
        t.mean += delta / t.n;
        t.m2 += delta * (v - t.mean);
    }
    t = warp_welford(t);
    if (lane == 0) scratch[warp] = t;
    __syncthreads();
    t = lane < blockDim.x / 32 ? scratch[lane] : Welford{0.0f, 0.0f, 0.0f};
    t = warp_welford(t);

    float rstd = rsqrtf(t.m2 / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = (row[i] - t.mean) * rstd * gamma[i] + beta[i];
    if (threadIdx.x == 0) {
        mean_out[blockIdx.x] = t.mean;
        rstd_out[blockIdx.x] = rstd;
    }
}
