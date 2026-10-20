NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17

all: saxpy

saxpy: saxpy.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: saxpy
	./saxpy

clean:
	rm -f saxpy

.PHONY: all run clean
