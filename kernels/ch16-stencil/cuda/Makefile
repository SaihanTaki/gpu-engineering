NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: conv2d

conv2d: conv2d.cu conv2d_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: conv2d
	./conv2d

clean:
	rm -f conv2d

.PHONY: all run clean
