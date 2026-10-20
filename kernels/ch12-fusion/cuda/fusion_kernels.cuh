// out = relu(a * x + y), as three elementwise kernels or one fused kernel.
// One thread per element.
#pragma once

__global__ void scale(const float *x, float *t, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) t[i] = a * x[i];
}

__global__ void add(const float *t, const float *y, float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = t[i] + y[i];
}

__global__ void relu(const float *u, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(u[i], 0.0f);
}

__global__ void fused(const float *x, const float *y, float *out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // v never leaves a register, so t and u never exist in DRAM.
        float v = a * x[i] + y[i];
        out[i] = fmaxf(v, 0.0f);
    }
}
