NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17

all: whoami

whoami: whoami.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: whoami
	./whoami

clean:
	rm -f whoami

.PHONY: all run clean
