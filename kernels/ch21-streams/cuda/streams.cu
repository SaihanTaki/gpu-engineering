// Copy in, compute, copy out: serially in the default stream, as a chunked
// pipeline over several streams (from pinned and from pageable memory), and as
// the same pipeline captured once into a CUDA Graph and replayed.
//
// Usage: ./streams [n] [streams] [chunks] [iters]
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "streams_kernels.cuh"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err_));                       \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

const int THREADS = 256;
const int MAX_STREAMS = 8;
const int REPS = 10;

struct Setup {
    int n, streams, chunks, iters;
    float *d;
    cudaStream_t s[MAX_STREAMS];
};

static void serial(const Setup &p, const float *in, float *out) {
    size_t bytes = (size_t)p.n * sizeof(float);
    // No stream argument: everything goes to the default stream, one after another.
    CUDA_CHECK(cudaMemcpy(p.d, in, bytes, cudaMemcpyHostToDevice));
    work<<<(p.n + THREADS - 1) / THREADS, THREADS>>>(p.d, p.n, p.iters);
    CUDA_CHECK(cudaMemcpy(out, p.d, bytes, cudaMemcpyDeviceToHost));
}

// Chunk c goes to stream c % streams; in, out must be pinned for the copies to overlap.
static void pipeline(const Setup &p, const float *in, float *out) {
    int chunk = (p.n + p.chunks - 1) / p.chunks;
    for (int c = 0; c < p.chunks; ++c) {
        int off = c * chunk;
        if (off >= p.n) break;
        int m = std::min(chunk, p.n - off);
        cudaStream_t st = p.s[c % p.streams];
        CUDA_CHECK(cudaMemcpyAsync(p.d + off, in + off, (size_t)m * sizeof(float), cudaMemcpyHostToDevice, st));
        work<<<(m + THREADS - 1) / THREADS, THREADS, 0, st>>>(p.d + off, m, p.iters);
        CUDA_CHECK(cudaMemcpyAsync(out + off, p.d + off, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, st));
    }
}

// Mean ms per call of run over REPS calls. Events in the legacy default stream
// wait for all blocking streams, so stop is reached only when every stream is done.
template <typename F>
static float time_ms(F run) {
    run();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < REPS; ++r) run();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

// Capture the pipeline into a graph. The legacy stream cannot be captured, so
// capture starts in s[0] and the other streams join it through events.
static cudaGraphExec_t capture(const Setup &p, const float *in, float *out) {
    cudaEvent_t fork, join[MAX_STREAMS];
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    for (int i = 0; i < p.streams; ++i) CUDA_CHECK(cudaEventCreateWithFlags(&join[i], cudaEventDisableTiming));

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(p.s[0], cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaEventRecord(fork, p.s[0]));
    for (int i = 1; i < p.streams; ++i) CUDA_CHECK(cudaStreamWaitEvent(p.s[i], fork, 0));
    pipeline(p, in, out);
    for (int i = 1; i < p.streams; ++i) {
        CUDA_CHECK(cudaEventRecord(join[i], p.s[i]));
        CUDA_CHECK(cudaStreamWaitEvent(p.s[0], join[i], 0));
    }
    CUDA_CHECK(cudaStreamEndCapture(p.s[0], &graph));

    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(fork));
    for (int i = 0; i < p.streams; ++i) CUDA_CHECK(cudaEventDestroy(join[i]));
    return exec;
}

static float cpu_work(float v, int iters) {
    for (int k = 0; k < iters; ++k) v = v * 0.999f + 0.001f;
    return v;
}

int main(int argc, char **argv) {
    long long nn = argc > 1 ? std::atoll(argv[1]) : (1 << 24);
    int streams = argc > 2 ? std::atoi(argv[2]) : 4;
    int chunks = argc > 3 ? std::atoi(argv[3]) : 16;
    int iters = argc > 4 ? std::atoi(argv[4]) : 100;
    if (nn < 1 || nn > INT_MAX / 4 || streams < 1 || streams > MAX_STREAMS || chunks < 1 || iters < 0) {
        std::fprintf(stderr, "usage: %s [n] [streams 1..%d] [chunks >= 1] [iters >= 0]\n", argv[0], MAX_STREAMS);
        return 1;
    }
    Setup p{};
    p.n = (int)nn;
    p.streams = streams;
    p.chunks = chunks;
    p.iters = iters;
    size_t bytes = (size_t)p.n * sizeof(float);

    int dev = 0, engines = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&engines, cudaDevAttrAsyncEngineCount, dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("%s, asyncEngineCount = %d\n", prop.name, engines);
    std::printf("n = %d floats (%.1f MiB), %d streams, %d chunks, %d iters\n", p.n, bytes / 1048576.0, streams, chunks,
                iters);

    float *in, *out, *ref;
    CUDA_CHECK(cudaMallocHost(&in, bytes));
    CUDA_CHECK(cudaMallocHost(&out, bytes));
    CUDA_CHECK(cudaMallocHost(&ref, bytes));
    std::vector<float> pg_in(p.n), pg_out(p.n);
    for (int i = 0; i < p.n; ++i) in[i] = pg_in[i] = (float)(i % 1000) / 500.0f - 1.0f;
    CUDA_CHECK(cudaMalloc(&p.d, bytes));
    for (int i = 0; i < streams; ++i) CUDA_CHECK(cudaStreamCreate(&p.s[i]));

    int failures = 0;
    float ms_serial = time_ms([&] { serial(p, in, ref); });
    for (int i = 0; i < p.n; ++i) failures += ref[i] != cpu_work(in[i], iters);
    std::printf("  %-18s %9.3f ms%s\n", "serial", ms_serial, failures ? "  MISMATCH vs CPU" : "");

    auto check = [&](const char *name, float ms, const float *got) {
        bool bad = std::memcmp(got, ref, bytes) != 0;
        failures += bad;
        std::printf("  %-18s %9.3f ms  %5.2fx vs serial%s\n", name, ms, ms_serial / ms, bad ? "  MISMATCH" : "");
    };

    std::memset(out, 0, bytes);
    check("streams, pinned", time_ms([&] { pipeline(p, in, out); }), out);

    // From pageable memory each cudaMemcpyAsync behaves synchronously, so nothing overlaps.
    check("streams, pageable", time_ms([&] { pipeline(p, pg_in.data(), pg_out.data()); }), pg_out.data());

    std::memset(out, 0, bytes);
    cudaGraphExec_t exec = capture(p, in, out);
    check("graph, pinned", time_ms([&] { CUDA_CHECK(cudaGraphLaunch(exec, p.s[0])); }), out);
    CUDA_CHECK(cudaGraphExecDestroy(exec));

    for (int i = 0; i < streams; ++i) CUDA_CHECK(cudaStreamDestroy(p.s[i]));
    CUDA_CHECK(cudaFree(p.d));
    CUDA_CHECK(cudaFreeHost(in));
    CUDA_CHECK(cudaFreeHost(out));
    CUDA_CHECK(cudaFreeHost(ref));

    if (failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
