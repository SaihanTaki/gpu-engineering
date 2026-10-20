NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: coalescing

coalescing: coalescing.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: coalescing
	./coalescing

clean:
	rm -f coalescing

.PHONY: all run clean
