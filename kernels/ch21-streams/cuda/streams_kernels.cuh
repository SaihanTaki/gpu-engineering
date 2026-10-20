// The per-chunk work: iters steps of v = 0.999 * v + 0.001 on each element.
// One thread per element. iters sets how long a chunk's kernel runs next to its copies.
#pragma once

__global__ void work(float *x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // _rn intrinsics keep the multiply and add separate, so the CPU check can match bit for bit.
        for (int k = 0; k < iters; ++k) v = __fadd_rn(__fmul_rn(v, 0.999f), 0.001f);
        x[i] = v;
    }
}
