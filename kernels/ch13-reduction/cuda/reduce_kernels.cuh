// Four sum reductions of float32. The first three reduce one block's slice of
// `in` in shared memory and write the block's sum to out[blockIdx.x]; the host
// launches them again on those partial sums until one value is left.
// reduce_atomic adds each block's sum into *out, so one launch covers the grid.
// blockDim.x must be a power of two, at least 64 for reduce_warp_final.
#pragma once

constexpr unsigned FULL_MASK = 0xffffffffu;

__global__ void reduce_interleaved(const float *in, float *out, unsigned n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        // The working threads are spread over every warp, so every warp diverges.
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_sequential(const float *in, float *out, unsigned n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
        // Threads 0 to s - 1 work: whole warps until s drops below 32.
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_warp_final(const float *in, float *out, unsigned n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        // The last 64 values fit in one warp: finish in registers, with no barrier.
        float v = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_down_sync(FULL_MASK, v, offset);
        }
        if (tid == 0) out[blockIdx.x] = v;
    }
}

// After offsets 16, 8, 4, 2 and 1, lane 0 holds the sum of all 32 lanes.
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

// *out must be zero before the launch.
__global__ void reduce_atomic(const float *in, float *out, unsigned n) {
    __shared__ float warp_sums[32];
    unsigned tid = threadIdx.x;
    unsigned lane = tid % 32;
    unsigned warp = tid / 32;
    float v = 0.0f;
    // Grid-stride loop: each thread sums many elements in a register first.
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        v += in[i];
    }
    v = warp_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
        v = warp_sum(v);
        // One atomic per block, not one per element.
        if (lane == 0) atomicAdd(out, v);
    }
}
