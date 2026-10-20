NVCC ?= nvcc
# --resource-usage prints each kernel's registers and shared memory at compile time.
NVCCFLAGS ?= -O2 -std=c++17 --resource-usage
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: occupancy

occupancy: occupancy.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $<

run: occupancy
	./occupancy

clean:
	rm -f occupancy

.PHONY: all run clean
