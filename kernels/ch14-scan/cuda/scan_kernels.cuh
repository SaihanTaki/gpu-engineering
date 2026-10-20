// Block-level prefix sums of int32 and the kernel that joins blocks together.
// Each scan kernel scans one tile of the input, writes the tile's total to
// block_sums (when it is not null), and writes an inclusive or exclusive scan.
#pragma once

constexpr int BLOCK = 256;
// Blelloch gives each thread two elements, so a block scans 512 of them.
constexpr int ITEMS = 2 * BLOCK;

// Hillis-Steele: log2(BLOCK) steps; step k adds the value 2^k places to the left.
template <bool INCLUSIVE>
__global__ void scan_hillis_steele(const int *in, int *out, int *block_sums, int n) {
    __shared__ int s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    s[tid] = i < n ? in[i] : 0;
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset *= 2) {
        // Read every left neighbour before anyone overwrites it.
        int left = tid >= offset ? s[tid - offset] : 0;
        __syncthreads();
        if (tid >= offset) {
            s[tid] += left;
        }
        __syncthreads();
    }

    if (block_sums && tid == BLOCK - 1) {
        block_sums[blockIdx.x] = s[BLOCK - 1];
    }
    if (i < n) {
        // The exclusive scan is the inclusive one shifted right by one.
        out[i] = INCLUSIVE ? s[tid] : (tid > 0 ? s[tid - 1] : 0);
    }
}

// Blelloch: an up-sweep reduction tree, then a down-sweep that hands each
// element the sum of everything to its left. 2 * (ITEMS - 1) adds in total.
template <bool INCLUSIVE>
__global__ void scan_blelloch(const int *in, int *out, int *block_sums, int n) {
    __shared__ int s[ITEMS];
    __shared__ int total;
    int tid = threadIdx.x;
    int base = blockIdx.x * ITEMS;
    // Two loads per thread, BLOCK apart, so each warp reads consecutive words.
    for (int k = tid; k < ITEMS; k += BLOCK) {
        s[k] = base + k < n ? in[base + k] : 0;
    }

    int offset = 1;
    for (int d = ITEMS / 2; d > 0; d /= 2) {
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            s[b] += s[a];
        }
        offset *= 2;
    }

    if (tid == 0) {
        total = s[ITEMS - 1];
        if (block_sums) {
            block_sums[blockIdx.x] = total;
        }
        s[ITEMS - 1] = 0;
    }

    for (int d = 1; d < ITEMS; d *= 2) {
        offset /= 2;
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            int t = s[a];
            s[a] = s[b];
            s[b] += t;
        }
    }
    __syncthreads();

    for (int k = tid; k < ITEMS; k += BLOCK) {
        if (base + k < n) {
            // s is the exclusive scan; the inclusive one is s shifted left, ending in the total.
            out[base + k] = !INCLUSIVE ? s[k] : (k + 1 < ITEMS ? s[k + 1] : total);
        }
    }
}

// offsets[b] is the sum of every tile before tile b: an exclusive scan of the block sums.
__global__ void add_offsets(int *out, const int *offsets, int n, int per_block) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += offsets[i / per_block];
    }
}
