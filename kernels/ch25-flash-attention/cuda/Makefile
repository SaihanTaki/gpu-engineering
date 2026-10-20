NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: flash_attention

flash_attention: flash_attention.cu flash_attention_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: flash_attention
	./flash_attention
	./flash_attention 1024 1

clean:
	rm -f flash_attention

.PHONY: all run clean
