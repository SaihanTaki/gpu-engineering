// x = 2 * x five ways: explicit copies from pageable and from pinned host memory,
// managed memory moved by page faults, managed memory moved by cudaMemPrefetchAsync,
// and plain malloc memory passed straight to the kernel (needs HMM or ATS).
// Uses the CUDA 13 forms of cudaMemPrefetchAsync and cudaMemAdvise, which take a cudaMemLocation.
#include <cstdio>
#include <cstdlib>
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

__global__ void scale(float *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.0f * x[i];
}

static const int THREADS = 256;

static void fill(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = (float)(i % 1000);
}

static bool check(const float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (x[i] != 2.0f * (float)(i % 1000)) {
            std::printf("mismatch at %zu: %f\n", i, x[i]);
            return false;
        }
    return true;
}

static int blocks_for(size_t n) { return (int)((n + THREADS - 1) / THREADS); }

struct Timer {
    cudaEvent_t start, stop;
    Timer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
    float end(cudaStream_t s = 0) {
        float ms;
        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

static void report(const char *name, size_t bytes, float ms, bool ok) {
    // Data crosses the link twice: in and back out.
    std::printf("  %-28s %9.3f ms  %7.2f GB/s  %s\n", name, ms, 2.0 * bytes / (ms * 1e-3) / 1e9,
                ok ? "OK" : "FAILED");
}

// Explicit copies. With pinned = false, h_x is pageable and the driver stages it.
static bool explicit_copy(size_t n, bool pinned) {
    size_t bytes = n * sizeof(float);
    float *h_x, *d_x;
    if (pinned) CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    else h_x = (float *)std::malloc(bytes);
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    fill(h_x, n);

    Timer t;
    t.begin();
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    scale<<<blocks_for(n), THREADS>>>(d_x, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
    float ms_gpu = t.end();

    // The CPU loop is timed in every run, so the managed runs compare fairly.
    Timer t2;
    t2.begin();
    bool ok = check(h_x, n);
    float ms_cpu = t2.end();
    report(pinned ? "cudaMemcpy, pinned" : "cudaMemcpy, pageable", bytes, ms_gpu + ms_cpu, ok);
    if (pinned) CUDA_CHECK(cudaFreeHost(h_x));
    else std::free(h_x);
    CUDA_CHECK(cudaFree(d_x));
    return ok;
}

// Managed memory. Without prefetch, the kernel's first touch of each page faults.
static bool managed(size_t n, int dev, bool prefetch) {
    size_t bytes = n * sizeof(float);
    float *x;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    fill(x, n);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaMemLocation gpu = {}, host = {};
    gpu.type = cudaMemLocationTypeDevice;
    gpu.id = dev;
    host.type = cudaMemLocationTypeHost;

    Timer t;
    t.begin(stream);
    if (prefetch) CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, gpu, 0, stream));
    scale<<<blocks_for(n), THREADS, 0, stream>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
    if (prefetch) CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, host, 0, stream));
    float ms_gpu = t.end(stream);

    // Without prefetch, this loop is where the pages fault back to the CPU.
    Timer t2;
    t2.begin();
    bool ok = check(x, n);
    float ms_cpu = t2.end();
    report(prefetch ? "managed + prefetch" : "managed, on-demand", bytes, ms_gpu + ms_cpu, ok);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x));
    return ok;
}

// Plain malloc memory used by the kernel directly. Only valid when pageableMemoryAccess is 1.
static bool system_allocated(size_t n, int dev) {
    size_t bytes = n * sizeof(float);
    float *x = (float *)std::malloc(bytes);
    fill(x, n);
    cudaMemLocation gpu = {};
    gpu.type = cudaMemLocationTypeDevice;
    gpu.id = dev;
    // A hint, not a requirement: the data will be read and written on this GPU.
    CUDA_CHECK(cudaMemAdvise(x, bytes, cudaMemAdviseSetPreferredLocation, gpu));

    Timer t;
    t.begin();
    scale<<<blocks_for(n), THREADS>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = check(x, n);
    float ms = t.end();
    report("malloc, HMM/ATS", bytes, ms, ok);
    std::free(x);
    return ok;
}

int main(int argc, char **argv) {
    size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (size_t)1 << 26;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    int concurrent, pageable;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev));
    std::printf("n = %zu floats (%.1f MiB), concurrentManagedAccess = %d, pageableMemoryAccess = %d\n", n,
                n * sizeof(float) / 1048576.0, concurrent, pageable);

    // The first launch loads the kernel, so it runs once untimed.
    float *warm;
    CUDA_CHECK(cudaMalloc(&warm, sizeof(float)));
    scale<<<1, 1>>>(warm, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(warm));

    bool ok = explicit_copy(n, false) && explicit_copy(n, true) && managed(n, dev, false);
    // cudaMemPrefetchAsync to a GPU needs concurrentManagedAccess.
    if (concurrent) ok = ok && managed(n, dev, true);
    else std::printf("  managed + prefetch: skipped, concurrentManagedAccess = 0\n");
    if (pageable) ok = ok && system_allocated(n, dev);
    else std::printf("  malloc, HMM/ATS: skipped, pageableMemoryAccess = 0\n");

    std::printf(ok ? "OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
