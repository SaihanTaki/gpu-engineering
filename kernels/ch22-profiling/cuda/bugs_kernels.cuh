// Five bugs and their fixes. Each pair computes the same thing; the buggy one
// is what compute-sanitizer is for. Device code only, so NVRTC can compile it.
#pragma once

constexpr int BLOCK = 256;

// 1. Lost updates (racecheck). Each block counts its positive elements in one
// shared counter and adds the block's count to the total.
__global__ void count_racy(const float *x, int n, unsigned int *total) {
    __shared__ unsigned int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // A load, an add and a store: another thread can load between them.
    if (i < n && x[i] > 0.0f) count += 1;
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(total, count);
}

__global__ void count_atomic(const float *x, int n, unsigned int *total) {
    __shared__ unsigned int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > 0.0f) atomicAdd(&count, 1u);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(total, count);
}

// 2. Out-of-bounds read (memcheck). Forward difference: out has n - 1 elements.
__global__ void diff_oob(const float *in, int n, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread n - 1 reads in[n], one past the end.
    if (i < n) out[i] = in[i + 1] - in[i];
}

__global__ void diff(const float *in, int n, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) out[i] = in[i + 1] - in[i];
}

// 3. Missing barrier (racecheck). Reverse each block of BLOCK elements;
// n must be a multiple of BLOCK.
__global__ void reverse_nosync(const float *in, float *out) {
    __shared__ float tile[BLOCK];
    int base = blockIdx.x * BLOCK;
    tile[threadIdx.x] = in[base + threadIdx.x];
    // No __syncthreads(): tile[BLOCK - 1 - t] belongs to another warp, which may not have written it yet.
    out[base + threadIdx.x] = tile[BLOCK - 1 - threadIdx.x];
}

__global__ void reverse(const float *in, float *out) {
    __shared__ float tile[BLOCK];
    int base = blockIdx.x * BLOCK;
    tile[threadIdx.x] = in[base + threadIdx.x];
    __syncthreads();
    out[base + threadIdx.x] = tile[BLOCK - 1 - threadIdx.x];
}

// 4. Uninitialized memory (initcheck). The kernel is correct; the host bug
// is a cudaMemset that clears n bytes instead of n ints.
__global__ void add_index(int *v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] += i;
}

// 5. Barrier in divergent code (synccheck). Sum of each block of x.
__global__ void block_sum_divergent(const float *x, int n, float *sums) {
    __shared__ float s[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    // Two barriers, one per branch. If n is not a multiple of 32, one warp of
    // the last block splits between them, and a warp must reach a barrier together.
    if (i < n) {
        s[threadIdx.x] = x[i];
        __syncthreads();
    } else {
        s[threadIdx.x] = 0.0f;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int k = 0; k < BLOCK; ++k) acc += s[k];
        sums[blockIdx.x] = acc;
    }
}

__global__ void block_sum(const float *x, int n, float *sums) {
    __shared__ float s[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    // One barrier that every thread reaches; threads past n contribute zero.
    s[threadIdx.x] = i < n ? x[i] : 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int k = 0; k < BLOCK; ++k) acc += s[k];
        sums[blockIdx.x] = acc;
    }
}
