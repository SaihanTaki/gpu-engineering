import type { Question } from './types';

export default [
	{
		q: 'On a typical PCIe system, host code reads `d_a[0]`, where `d_a` came from `cudaMalloc`. What happens?',
		options: [
			'The read returns a stale copy of the value from host RAM',
			'The program crashes, because no CPU memory is mapped at that address',
			'The driver fetches the value over PCIe and returns it',
			'The read returns zero until the first kernel writes to `d_a`',
		],
		answer: 1,
		why: 'A pointer from `cudaMalloc` is an address in GPU memory, which the CPU cannot read through. Nothing copies the value for you; that is what `cudaMemcpy` is for.',
	},
	{
		q: 'Right after `cudaMalloc(&d_a, bytes)` succeeds, what does the allocation hold?',
		options: [
			'Zeros, because the driver clears it',
			'A copy of the host array `h_a`',
			'The output of the previous kernel',
			'No useful data yet',
		],
		answer: 3,
		why: '`cudaMalloc` only reserves memory. Data gets there with a host-to-device `cudaMemcpy` or a kernel that writes it.',
	},
	{
		q: 'When does a device-to-host `cudaMemcpy` return?',
		options: [
			'Once the data has arrived in host memory',
			'As soon as the copy is queued',
			'Once the source can be reused, possibly before the data arrives',
			'Only after the next kernel launch',
		],
		answer: 0,
		why: 'A device-to-host copy returns once the data has arrived. The "source can be reused" rule describes a host-to-device copy from ordinary memory.',
	},
	{
		q: 'Vector add on 2^24 floats copies `a` and `b` in and `c` out. How many bytes cross PCIe in total?',
		options: ['64 MiB', '128 MiB', '192 MiB', '256 MiB'],
		answer: 2,
		why: 'Each array is 2^24 x 4 bytes = 64 MiB, and three arrays cross the link. 128 MiB counts only the copy in.',
	},
	{
		q: 'On the A100, the copies take far longer than the kernel in the simulation. Why?',
		options: [
			'The copies move more bytes than the kernel does',
			'`cudaMemcpy` runs on a single CPU thread',
			'Both move the same bytes, but GPU memory is about 64 times faster than the PCIe link',
			'A floating point addition is slow compared with a copy',
		],
		answer: 2,
		why: 'The kernel reads `a` and `b` and writes `c`, the same 192 MiB the copies move, at 2,039 GB/s instead of 32 GB/s. Vector add does one addition per 12 bytes, so the arithmetic barely counts.',
	},
	{
		q: 'You switch the simulation from the A100 to the H100, whose PCIe link is twice as fast. What happens to the share of time spent on copies?',
		options: [
			'It stays above 98%, because the H100 memory is faster too',
			'It halves, because the link is twice as fast',
			'It drops to about 75%',
			'The kernel becomes the longest bar',
		],
		answer: 0,
		why: 'Both bandwidths grow, so the ratio between them barely moves. The copies are still about 98% of the total.',
	},
	{
		q: 'In the simulation, you raise `n` from 2^20 to 2^28. What happens to the share of time spent on copies?',
		options: [
			'It grows, because bigger arrays take longer to copy',
			'It shrinks, because the kernel time grows faster than the copies',
			'It grows, because the kernel is cheaper per element at large sizes',
			'It stays the same, because every phase grows in proportion to `n`',
		],
		answer: 3,
		why: 'The copy time and the kernel time are both bytes divided by a bandwidth, and the bytes scale with `n`. Fixed costs per call are left out of the simulation.',
	},
	{
		q: 'What is the chapter\'s rule for real programs that use the GPU?',
		options: [
			'Move every vector add to the GPU, however small',
			'Copy data in once, keep it on the GPU across many kernels, and copy back only the final result',
			'Copy each kernel\'s output back to the host to check it before the next kernel',
			'Use smaller blocks so that the copies finish sooner',
		],
		answer: 1,
		why: 'The copies are the expensive part. A chain of kernels on data that already lives on the GPU pays for the copies once.',
	},
	{
		q: 'Why does a real `cudaMemcpy` from ordinary (pageable) host memory run slower than the PCIe peak?',
		options: [
			'The driver first copies the data into a staging buffer',
			'The link can only carry one array at a time',
			'GPU memory accepts writes at half its read bandwidth',
			'The data is compressed before it crosses the link',
		],
		answer: 0,
		why: 'Pageable memory goes through a staging buffer first. Pinned host memory avoids that step, as chapter 10 shows.',
	},
	{
		q: 'With `threads = 256` and `blocks = (n + threads - 1) / threads`, how many blocks launch for `n = 1,000,000`?',
		options: ['3,906', '4,096', '3,907', '3,908'],
		answer: 2,
		why: '1,000,000 / 256 is 3,906.25, and the formula rounds up to 3,907. Plain integer division would give 3,906 and leave the last 64 elements without a thread.',
	},
	{
		q: 'With 256 threads per block and a rounded-up grid, `n = 5,000`. How many threads fail the check `i < n`?',
		options: ['0', '120', '136', '256'],
		answer: 1,
		why: '5,000 / 256 rounds up to 20 blocks, which is 5,120 threads. The extra 120 are all in the last block.',
	},
	{
		q: 'Why is 256 threads per block a common starting point?',
		options: [
			'It is the largest block size CUDA allows',
			'It matches the number of CUDA cores in one SM',
			'The PCIe link transfers data in units of 256 elements',
			'It is a multiple of the 32-thread warp and well under the 1,024 limit',
		],
		answer: 3,
		why: 'The limit is 1,024 threads per block, not 256. Chapter 5 shows how to choose the block size properly.',
	},
	{
		q: 'A kernel is launched with 2,048 threads per block. Where does the error show up?',
		options: [
			'In the return value of the `<<<...>>>` launch statement',
			'As an nvcc compile error',
			'In the next `cudaFree` call, and nowhere earlier',
			'In `cudaGetLastError()` called right after the launch',
		],
		answer: 3,
		why: 'A launch returns nothing, so errors in the launch itself, such as a block over 1,024 threads, are read with `cudaGetLastError()` right after it.',
	},
	{
		q: 'An out-of-bounds access inside a kernel is usually not reported by `cudaGetLastError()` right after the launch. Why?',
		options: [
			'The launch is asynchronous and returns before the kernel has run',
			'`cudaGetLastError()` only reports errors from host code',
			'Out-of-bounds accesses on the GPU are never reported',
			'The `CUDA_CHECK` macro clears the error first',
		],
		answer: 0,
		why: 'An error that happens while the kernel runs can only be reported after it runs. `cudaDeviceSynchronize()` is the call that guarantees that.',
	},
	{
		q: 'A kernel makes an illegal memory access. What do later CUDA calls in the same process return?',
		options: [
			'Success, once `cudaGetLastError()` has reset the error',
			'Success, once `cudaDeviceSynchronize()` has returned',
			'The same error, because the GPU context is unusable',
			'Success for copies, an error only for kernel launches',
		],
		answer: 2,
		why: 'An illegal memory access is a sticky error, so `cudaGetLastError()` cannot reset it. That is why the macro exits instead of carrying on.',
	},
	{
		q: 'The CUDA program launches the kernel once, untimed, before the timed phases. Why?',
		options: [
			'To fill the L2 cache with `a` and `b`',
			'Because the first launch includes loading the kernel onto the GPU',
			'To allocate the device arrays',
			'To check the result before the copies',
		],
		answer: 1,
		why: 'The first launch pays a one-time cost to load the kernel. Timing it would add that cost to the kernel phase.',
	},
	{
		q: 'In Numba, what happens when you pass NumPy arrays straight to a kernel?',
		options: [
			'Numba copies each array to the device before the launch and back after it, every time',
			'Numba copies back only the arrays the kernel writes',
			'The kernel reads host memory directly over PCIe',
			'Numba raises an exception, because it accepts only device arrays',
		],
		answer: 0,
		why: 'Numba copies every array both ways on each launch, even the ones the kernel only reads. `cuda.to_device` and `copy_to_host` let you control the copies.',
	},
	{
		q: 'In the Triton version, what does the job of the `if (i < n)` bounds check?',
		options: [
			'`triton.cdiv(n, BLOCK)`',
			'`torch.empty_like(a)`',
			'Nothing, because the Triton grid always matches `n` exactly',
			'The mask on the loads and the store',
		],
		answer: 3,
		why: '`triton.cdiv` is the rounded-up grid, which is what creates the extra elements in the first place. The mask stops the program from touching them.',
	},
	{
		q: 'You have no NVIDIA GPU. How can you still run the Python versions of the vector add?',
		options: [
			'Build the CUDA version with `ARCH=75`',
			'Pass a small count to `./cuda/vector_add`',
			'Set `NUMBA_ENABLE_CUDASIM=1` or `TRITON_INTERPRET=1` and use a small count',
			'Run `make -C cuda run` with the CPU backend',
		],
		answer: 2,
		why: 'Both settings run the kernels on the CPU. Their times then say nothing about GPUs.',
	},
] satisfies Question[];
