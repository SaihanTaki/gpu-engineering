// Two histograms of n bytes into BINS = 256 counters.
// Both use a grid-stride loop, so any grid size covers any n.
#pragma once

constexpr int BINS = 256;

// Every element is one atomic on global memory. Equal bytes queue on one counter.
__global__ void hist_global(const unsigned char *data, int n, unsigned int *hist) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

// Each block counts into its own copy in shared memory, then adds that copy to
// the global histogram: at most BINS global atomics per block.
__global__ void hist_shared(const unsigned char *data, int n, unsigned int *hist) {
    __shared__ unsigned int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        local[b] = 0;
    }
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Only threads of this block compete for this copy.
        atomicAdd(&local[data[i]], 1u);
    }
    // Every count must be final before it is merged.
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        if (local[b] > 0) {
            atomicAdd(&hist[b], local[b]);
        }
    }
}
