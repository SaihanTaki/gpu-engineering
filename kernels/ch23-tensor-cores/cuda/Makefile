NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# WMMA needs sm_70 or later. sm_80 is an A100; use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: wmma_gemm

wmma_gemm: wmma_gemm.cu wmma_gemm.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: wmma_gemm
	./wmma_gemm

clean:
	rm -f wmma_gemm

.PHONY: all run clean
