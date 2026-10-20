NVCC ?= nvcc
# -lineinfo lets compute-sanitizer and Nsight Compute name the source line; -rdynamic keeps host frames.
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo -Xcompiler -rdynamic
# sm_80 is an A100. Use ARCH=75 for a T4, 90 for an H100.
ARCH ?= 80
SANITIZER ?= compute-sanitizer

all: bugs

# NVTX 3 is header-only; Nsight Systems' guide asks for -ldl.
bugs: bugs.cu bugs_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(ARCH) -o $@ $< -ldl

run: bugs
	./bugs all

# --padding puts a guard after every allocation, so a read one past the end
# cannot land silently in the next buffer.
memcheck: bugs
	$(SANITIZER) --tool memcheck --padding 32 ./bugs oob
	$(SANITIZER) --tool memcheck ./bugs launch

racecheck: bugs
	$(SANITIZER) --tool racecheck ./bugs race
	$(SANITIZER) --tool racecheck ./bugs nosync

initcheck: bugs
	$(SANITIZER) --tool initcheck ./bugs uninit

# The programming guide says a barrier in divergent code may hang, hence the timeout.
synccheck: bugs
	timeout 120 $(SANITIZER) --tool synccheck ./bugs divergent

sanitize: memcheck racecheck initcheck synccheck

nsys: bugs
	nsys profile --trace=cuda,nvtx --stats=true -o bugs-timeline ./bugs all

ncu: bugs
	ncu --kernel-name regex:count_ --set full -o bugs-count ./bugs race

clean:
	rm -f bugs bugs-timeline.* bugs-count.ncu-rep

.PHONY: all run memcheck racecheck initcheck synccheck sanitize nsys ncu clean
