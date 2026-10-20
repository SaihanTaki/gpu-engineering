NVCC ?= nvcc
# -Xptxas -v makes ptxas print registers, stack frame and spill bytes per kernel.
NVCCFLAGS ?= -O2 -std=c++17 -Xptxas -v
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: memory_spaces

memory_spaces: memory_spaces.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: memory_spaces
	./memory_spaces

clean:
	rm -f memory_spaces

.PHONY: all run clean
