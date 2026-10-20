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

constexpr int THREADS = 256;

// Lane k of a warp reads element k + offset: offset 0 touches 4 sectors per warp, any offset that is not a multiple of 8 touches 5.
__global__ void copy_offset(const float *in, float *out, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i + offset];
}

// Neighbouring lanes read elements stride apart; from stride 8 on, every lane needs its own sector.
__global__ void copy_strided(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[static_cast<size_t>(i) * stride];
}

struct Particle {
    float x, y, z;
};

// Array of structs: reading only x is a stride-3 load, 12 sectors per warp.
__global__ void get_x_aos(const Particle *p, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = p[i].x;
}

// Struct of arrays: the x values sit next to each other, 4 sectors per warp.
__global__ void get_x_soa(const float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

static float *to_device(const std::vector<float> &h) {
    float *d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// Times launch() with CUDA events, averaged over a few runs after one warm-up.
template <typename F>
static float time_ms(F launch) {
    constexpr int REPS = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

// Counts output elements that differ from expected, and prints the useful bandwidth: bytes read and written by the threads, per second.
static int check(const char *label, float *d_out, const std::vector<float> &expected, float ms) {
    int n = static_cast<int>(expected.size());
    std::vector<float> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) errors += out[i] != expected[i];
    double gb = 2.0 * n * sizeof(float) / 1e9;
    std::printf("%-16s %8.3f ms  %7.1f GB/s useful\n", label, ms, gb / (ms / 1e3));
    return errors;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    constexpr int MAX_STRIDE = 32;
    int blocks = (n + THREADS - 1) / THREADS;

    // Enough input for the largest stride; in[j] = j makes every expected value easy to state.
    std::vector<float> in(static_cast<size_t>(n) * MAX_STRIDE);
    for (size_t j = 0; j < in.size(); ++j) in[j] = static_cast<float>(j % 1000003);
    float *d_in = to_device(in);
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    std::vector<float> expected(n);
    int errors = 0;
    char label[32];

    for (int offset : {0, 1, 8, 16}) {
        for (int i = 0; i < n; ++i) expected[i] = in[i + offset];
        float ms = time_ms([&] { copy_offset<<<blocks, THREADS>>>(d_in, d_out, n, offset); });
        std::snprintf(label, sizeof label, "offset %d", offset);
        errors += check(label, d_out, expected, ms);
    }
    for (int stride : {1, 2, 4, 8, 32}) {
        for (int i = 0; i < n; ++i) expected[i] = in[static_cast<size_t>(i) * stride];
        float ms = time_ms([&] { copy_strided<<<blocks, THREADS>>>(d_in, d_out, n, stride); });
        std::snprintf(label, sizeof label, "stride %d", stride);
        errors += check(label, d_out, expected, ms);
    }

    std::vector<Particle> aos(n);
    std::vector<float> xs(n);
    for (int i = 0; i < n; ++i) {
        aos[i] = {static_cast<float>(i), -1.0f, -2.0f};
        xs[i] = static_cast<float>(i);
        expected[i] = static_cast<float>(i);
    }
    Particle *d_aos;
    CUDA_CHECK(cudaMalloc(&d_aos, n * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_aos, aos.data(), n * sizeof(Particle), cudaMemcpyHostToDevice));
    float *d_x = to_device(xs);
    errors += check("x from AoS", d_out, expected, time_ms([&] { get_x_aos<<<blocks, THREADS>>>(d_aos, d_out, n); }));
    errors += check("x from SoA", d_out, expected, time_ms([&] { get_x_soa<<<blocks, THREADS>>>(d_x, d_out, n); }));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_x));
    if (errors != 0) {
        std::printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
