#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// y = a * x + y. The grid-stride loop lets any grid size cover all n elements.
__global__ void saxpy(int n, float a, const float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// At most 256 threads per block, and at least 8 such blocks resident per SM:
// the compiler must fit each thread in 65536 / (8 * 256) = 32 registers.
__global__ void __launch_bounds__(256, 8)
saxpy_bounded(int n, float a, const float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

static int warps_per_sm(const cudaDeviceProp &p) {
    return p.maxThreadsPerMultiProcessor / p.warpSize;
}

// Prints what the compiler gave the kernel and what the occupancy API makes of it.
static void report(const char *name, const void *kernel, const cudaDeviceProp &p) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    std::printf("kernel %s regs=%d static_smem=%zu max_threads=%d\n", name, attr.numRegs,
                attr.sharedSizeBytes, attr.maxThreadsPerBlock);

    // Above 48 KB of dynamic shared memory a kernel must opt in, even just to ask.
    int optin = (int)(p.sharedMemPerBlockOptin - attr.sharedSizeBytes);
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));

    const size_t smems[] = {0, 16 * 1024, 48 * 1024, 100 * 1024};
    const int sizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t smem : smems) {
        if (smem > (size_t)optin) continue;
        for (int block : sizes) {
            if (block > attr.maxThreadsPerBlock) continue;
            int blocks = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, block, smem));
            double occ = (double)blocks * ((block + p.warpSize - 1) / p.warpSize) / warps_per_sm(p);
            std::printf("occupancy kernel=%s smem=%zu block=%d blocks=%d theoretical=%.1f%%\n", name,
                        smem, block, blocks, 100.0 * occ);
        }
    }
}

static bool check(const std::vector<float> &x, const std::vector<float> &y0,
                  const std::vector<float> &y, float a) {
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::fabs(y[i] - (a * x[i] + y0[i])) > 1e-5f) {
            std::fprintf(stderr, "mismatch at %zu: %f\n", i, y[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
    const float a = 2.0f;

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("device %s cc=%d.%d sms=%d max_warps=%d regs=%d smem_per_sm=%zu reserved=%zu\n",
                p.name, p.major, p.minor, p.multiProcessorCount, warps_per_sm(p),
                p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor, p.reservedSharedMemPerBlock);

    report("saxpy", (const void *)saxpy, p);
    report("saxpy_bounded", (const void *)saxpy_bounded, p);

    // The block size the runtime suggests for the highest occupancy, and the
    // smallest grid that reaches it on every SM.
    int min_grid = 0, block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, saxpy, 0, 0));
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, saxpy, block, 0));
    std::printf("suggested block=%d min_grid=%d blocks_per_sm=%d\n", block, min_grid, per_sm);

    std::vector<float> h_x(n), h_y0(n), h_y(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = (i % 1000) * 0.001f;
        h_y0[i] = 1.0f - (i % 7) * 0.25f;
    }
    size_t bytes = (size_t)n * sizeof(float);
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    // One full wave: as many blocks as can be resident on all SMs at once.
    int grid = per_sm * p.multiProcessorCount;
    CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), bytes, cudaMemcpyHostToDevice));
    saxpy<<<grid, block>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    bool ok = check(h_x, h_y0, h_y, a);

    CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), bytes, cudaMemcpyHostToDevice));
    saxpy_bounded<<<8 * p.multiProcessorCount, 256>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    ok = check(h_x, h_y0, h_y, a) && ok;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
