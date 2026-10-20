"""Copy in, compute, copy out: serially, and as a chunked pipeline over several streams.

Usage: python numba/streams.py [n] [streams] [chunks] [iters]

Under NUMBA_ENABLE_CUDASIM=1 every call runs synchronously on the CPU, so the
results are checked but nothing overlaps and the times mean nothing.
"""

import sys
import time

import numpy as np
from numba import cuda

THREADS = 256
A = np.float32(0.999)
B = np.float32(0.001)


@cuda.jit
def work(x, iters):
    i = cuda.grid(1)
    if i < x.size:
        v = x[i]
        for _ in range(iters):
            v = v * A + B
        x[i] = v


def reference(x, iters):
    v = x.astype(np.float32, copy=True)
    for _ in range(iters):
        v = v * A + B
    return v


def blocks(n):
    return (n + THREADS - 1) // THREADS


def serial(x, iters):
    # No stream: the copies and the kernel go to the default stream, in order.
    d = cuda.to_device(x)
    work[blocks(d.size), THREADS](d, iters)
    return d.copy_to_host()


def serial_into(h_in, h_out, d, iters):
    """The serial run on preallocated buffers, so its time leaves out allocation like the pipeline's."""
    d.copy_to_device(h_in)
    work[blocks(d.size), THREADS](d, iters)
    d.copy_to_host(h_out)


def pipeline(h_in, h_out, d, streams, chunks, iters):
    """Chunk c goes to streams[c % len(streams)]. h_in and h_out should be pinned."""
    n = h_in.size
    size = -(-n // chunks)
    for c in range(chunks):
        lo, hi = c * size, min(n, (c + 1) * size)
        if lo >= hi:
            break
        st = streams[c % len(streams)]
        d[lo:hi].copy_to_device(h_in[lo:hi], stream=st)
        work[blocks(hi - lo), THREADS, st](d[lo:hi], iters)
        d[lo:hi].copy_to_host(h_out[lo:hi], stream=st)


def streamed(x, n_streams, chunks, iters):
    h_in = cuda.pinned_array(x.shape, dtype=np.float32)
    h_out = cuda.pinned_array(x.shape, dtype=np.float32)
    h_in[:] = x
    d = cuda.device_array(x.shape, dtype=np.float32)
    streams = [cuda.stream() for _ in range(n_streams)]
    pipeline(h_in, h_out, d, streams, chunks, iters)
    # Wait for these streams only; cuda.synchronize() would wait for the whole device.
    for st in streams:
        st.synchronize()
    return np.array(h_out)


def time_ms(run, reps=10):
    run()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / reps


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 24
    n_streams = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    x = ((np.arange(n) % 1000) / 500.0 - 1.0).astype(np.float32)
    print(f"n = {n} floats ({4 * n / 2**20:.1f} MiB), {n_streams} streams, {chunks} chunks, {iters} iters")

    ref = serial(x, iters)
    ok = np.allclose(ref, reference(x, iters), rtol=1e-5, atol=1e-6)

    h_in = cuda.pinned_array(x.shape, dtype=np.float32)
    h_out = cuda.pinned_array(x.shape, dtype=np.float32)
    h_in[:] = x
    d = cuda.device_array(x.shape, dtype=np.float32)
    streams = [cuda.stream() for _ in range(n_streams)]

    # Events time GPU work between two points in a stream; this pair brackets the whole run.
    start, stop = cuda.event(), cuda.event()
    start.record(streams[0])
    for st in streams[1:]:
        start.wait(st)
    pipeline(h_in, h_out, d, streams, chunks, iters)
    for st in streams[1:]:
        done = cuda.event()
        done.record(st)
        done.wait(streams[0])
    stop.record(streams[0])
    stop.synchronize()
    same = np.array_equal(np.array(h_out), ref)
    print(f"  one pipelined run, events: {start.elapsed_time(stop):9.3f} ms")

    h_ser = cuda.pinned_array(x.shape, dtype=np.float32)
    ms_serial = time_ms(lambda: serial_into(h_in, h_ser, d, iters))
    ms_streams = time_ms(lambda: pipeline(h_in, h_out, d, streams, chunks, iters))
    print(f"  serial                     {ms_serial:9.3f} ms")
    print(f"  streams, pinned            {ms_streams:9.3f} ms  {ms_serial / ms_streams:5.2f}x vs serial")
    print("OK" if ok and same else "FAILED")
    return 0 if ok and same else 1


if __name__ == "__main__":
    sys.exit(main())
