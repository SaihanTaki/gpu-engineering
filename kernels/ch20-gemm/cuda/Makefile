NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: gemm

gemm: gemm.cu gemm_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $< -lcublas

run: gemm
	./gemm

clean:
	rm -f gemm

.PHONY: all run clean
