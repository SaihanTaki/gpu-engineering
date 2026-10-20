NVCC ?= nvcc
# Host code must not fuse v * 0.999f + 0.001f into an FMA, or the CPU check would differ from the GPU.
NVCCFLAGS ?= -O2 -std=c++17 -Xcompiler -ffp-contract=off
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: streams

streams: streams.cu streams_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: streams
	./streams

clean:
	rm -f streams

.PHONY: all run clean
