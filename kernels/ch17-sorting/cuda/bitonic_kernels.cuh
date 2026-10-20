// Bitonic sort of m floats, ascending, m a power of two.
// Stage (k, j) pairs key i with key i + j for every i whose bit j is clear.
// The pair is put in ascending order when bit k of i is clear, else descending.
#pragma once

constexpr int THREADS = 256;            // threads per block for bitonic_step
constexpr int TILE = 1024;              // keys per block in bitonic_shared
constexpr int TILE_THREADS = TILE / 2;  // one thread per comparator

__device__ inline void compare_swap(float &a, float &b, bool up) {
    if ((a > b) == up) {
        float t = a;
        a = b;
        b = t;
    }
}

// Lower key of comparator t in a stage of distance j: skip the upper half of every 2j group.
__device__ inline int lower_key(int t, int j) { return (t / j) * 2 * j + t % j; }

// One stage over the whole array, one thread per comparator. The host launches it once per stage.
__global__ void bitonic_step(float *a, int m, int j, int k) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m / 2) return;
    int i = lower_key(t, j);
    compare_swap(a[i], a[i + j], (i & k) == 0);
}

// Every stage with j < TILE for k = k_first, 2 k_first, ..., k_last, on one tile in shared memory.
// Those stages never pair keys from different tiles, so blocks need no global barrier.
__global__ void bitonic_shared(float *a, int k_first, int k_last) {
    __shared__ float s[TILE];
    int base = blockIdx.x * TILE;
    int t = threadIdx.x;
    s[t] = a[base + t];
    s[t + TILE_THREADS] = a[base + t + TILE_THREADS];
    __syncthreads();
    for (int k = k_first; k <= k_last; k *= 2) {
        for (int j = min(k, TILE) / 2; j > 0; j /= 2) {
            int i = lower_key(t, j);
            // The direction depends on the global index, so neighbouring tiles sort opposite ways.
            compare_swap(s[i], s[i + j], ((base + i) & k) == 0);
            __syncthreads();
        }
    }
    a[base + t] = s[t];
    a[base + t + TILE_THREADS] = s[t + TILE_THREADS];
}
