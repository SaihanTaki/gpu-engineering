NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
# The listings in ../generated target sm_80 (A100). Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80

all: saxpy

saxpy: saxpy.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: saxpy
	./saxpy

# PTX for the virtual architecture compute_$(ARCH).
ptx: saxpy.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_$(ARCH) -ptx -o saxpy.ptx $<

# Machine code for the real architecture sm_$(ARCH), printed as SASS.
sass: saxpy.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_$(ARCH) -code=sm_$(ARCH) -cubin -o saxpy.cubin $<
	cuobjdump -sass saxpy.cubin

# An executable whose fatbin holds both sm_$(ARCH) SASS and compute_$(ARCH) PTX.
fatbin: saxpy.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_$(ARCH) -code=sm_$(ARCH),compute_$(ARCH) -o saxpy_fat $<
	cuobjdump -lelf saxpy_fat
	cuobjdump -lptx saxpy_fat

clean:
	rm -f saxpy saxpy_fat saxpy.ptx saxpy.cubin

.PHONY: all run ptx sass fatbin clean
