NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: scan

scan: scan.cu scan_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: scan
	./scan

clean:
	rm -f scan

.PHONY: all run clean
