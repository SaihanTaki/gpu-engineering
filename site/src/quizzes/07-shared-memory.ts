import type { Question } from './types';

export default [
	{
		q: 'Two 1024 x 1024 float matrices are multiplied naively, one thread per output element. How many global loads does each thread issue?',
		options: ['1,024', '2,048', '4,096', '1,048,576'],
		answer: 1,
		why: 'Each thread reads a row of A and a column of B, 2n = 2,048 loads. 1,024 counts only one of the two operands.',
	},
	{
		q: 'Which statement about the scope and lifetime of shared memory is correct?',
		options: [
			'One copy per SM, shared by every block resident on that SM at the time',
			'One copy per kernel launch, visible to every block',
			'One copy per block, which exists only while that block runs',
			'One copy per warp, kept until the kernel ends',
		],
		answer: 2,
		why: 'Every block gets its own copy, and it disappears when the block finishes. That is why a block must fill it before reading and cannot pass data to the next block through it.',
	},
	{
		q: 'Shared memory is carved from the same on-chip storage as which other memory?',
		options: ['The register file', 'The L2 cache', 'Device memory', 'The L1 cache'],
		answer: 3,
		why: 'Each SM has one block of storage split between L1 and shared memory. L2 is shared by all SMs and sits in front of device memory, not inside the SM.',
	},
	{
		q: 'A block needs 64 KB of shared memory. What does the CUDA C++ kernel have to do?',
		options: [
			'Use `extern __shared__`, pass the bytes at launch, and call `cudaFuncSetAttribute`',
			'Declare `__shared__ float buf[16384]`, since static shared arrays have no size limit at all',
			'Allocate the buffer with `cudaMalloc` and pass the pointer as a kernel argument',
			'Split the array into two static `__shared__` arrays of 32 KB each',
		],
		answer: 0,
		why: 'Static shared memory is limited to 48 KB per block, and two static arrays count against the same limit. Going past it needs the dynamic form and the `cudaFuncAttributeMaxDynamicSharedMemorySize` opt in.',
	},
	{
		q: 'In Numba, where does the byte count for dynamic shared memory go in the launch configuration?',
		options: [
			'`kernel[blocks, threads, bytes]`',
			'`kernel[blocks, bytes, threads]`',
			'`kernel[bytes, blocks, threads]`',
			'`kernel[blocks, threads, 0, bytes]`',
		],
		answer: 3,
		why: 'The bytes go in the fourth slot, after the stream. The CUDA C++ launch puts them third, which is the tempting mix up.',
	},
	{
		q: 'In the tiled kernel, the first `__syncthreads()` sits between loading `As` and `Bs` and the inner loop. What can go wrong without it?',
		options: [
			'Two threads write the same slot of `As` at once and one of the two writes is lost',
			'A thread reads a slot of `As` or `Bs` that another thread has not written yet',
			'The global loads stop being coalesced',
			'The compiler refuses to unroll the inner loop',
		],
		answer: 1,
		why: 'Warps run whenever their next instruction is ready, so nothing says another warp has finished its store. Each thread writes its own slot, so no write is lost.',
	},
	{
		q: 'What is the barrier at the end of each phase, after the inner loop, for?',
		options: [
			'It makes the sum in each thread visible to the other threads',
			'It flushes `As` and `Bs` to global memory',
			'It stops a thread overwriting a tile another thread is still reading',
			'It waits for the global loads of the next phase to arrive in shared memory',
		],
		answer: 2,
		why: 'This is the after reading, before overwriting case. Without it a warp can start loading the next tiles while others still need the current ones.',
	},
	{
		q: 'When may `__syncthreads()` appear inside an `if`?',
		options: [
			'Only when the condition evaluates the same way for every thread of the block',
			'Only when the condition evaluates the same way for every thread of a warp',
			'Never, the call must be at the top level of the kernel',
			'Always, threads that skip it are released automatically',
		],
		answer: 0,
		why: 'The guide allows it only if the condition is identical across the whole block, otherwise the code is likely to hang. A warp is not enough, because the barrier waits for every thread of the block.',
	},
	{
		q: 'With `n = 1024` and `TILE = 16`, how many global loads does each thread of the tiled kernel issue?',
		options: ['64', '256', '128', '2,048'],
		answer: 2,
		why: 'Each thread issues `2n / TILE` = 2,048 / 16 = 128 loads. 2,048 is the naive count, and 64 forgets that both A and B are loaded.',
	},
	{
		q: 'For the whole grid, by what factor does tiling with `TILE = 32` cut the number of global loads?',
		options: ['2', '16', '32', '1,024'],
		answer: 2,
		why: 'The grid issues `2n³` loads without shared memory and `2n³ / TILE` with it, so the ratio is `TILE`. 16 is the ratio for the 16 x 16 blocks used in the code.',
	},
	{
		q: 'How many values does each thread read for its dot product in the tiled kernel?',
		options: [
			'`2n / TILE`, the same as its number of global loads',
			'`2n`, but most of them come from shared memory',
			'`n`, because each element of A is read once',
			'`TILE`, one per term of a phase',
		],
		answer: 1,
		why: 'The number of reads does not change; only where they come from does. `2n / TILE` counts global loads, not reads.',
	},
	{
		q: 'How many bytes of shared memory do the two tiles take per block at `TILE = 32`?',
		options: ['4,096', '2,048', '16,384', '8,192'],
		answer: 3,
		why: 'Two tiles of `TILE²` floats: 2 x 32² x 4 = 8,192 bytes. 2,048 is the figure for `TILE = 16`.',
	},
	{
		q: 'With one thread per output, why can `TILE` not be 64?',
		options: [
			'Its two tiles would need more than 48 KB of shared memory',
			'Its two tiles would need more shared memory than an SM has',
			'Its column reads would always cause bank conflicts',
			'Its block would need more than 1,024 threads',
		],
		answer: 3,
		why: 'A 64 x 64 block is 4,096 threads, over the limit of 1,024. Its tiles take only 32 KB, which would fit.',
	},
	{
		q: 'Why use shared memory instead of relying on L1 to catch the repeated loads?',
		options: [
			'L1 does not cache global loads on these GPUs',
			'L1 is slower than device memory when the same address is read repeatedly',
			'Shared memory holds exactly what the block put there, a cache does not',
			'Shared memory is larger than L1 on every GPU in the course',
		],
		answer: 2,
		why: 'The chapter notes that some repeated loads do hit in L1 or L2. The difference is control: the code decides what stays in shared memory and for how long.',
	},
	{
		q: 'The chapter kernels handle `n` that is not a multiple of 16. What do threads past the edge of the matrix do?',
		options: [
			'They load zeros and stay in the loop so they reach both barriers',
			'They return at the start of the kernel',
			'They skip the loads but still run the inner loop',
			'They are never launched, because the grid is rounded down',
		],
		answer: 0,
		why: 'An early return would leave barriers that not every thread reaches, and slots of `As` and `Bs` that nobody writes. Loading zeros keeps the sum correct.',
	},
	{
		q: 'The Numba kernel PTX shows 32 `ld.shared.f32` instructions. Where do they come from?',
		options: [
			'One per thread of a warp',
			'The unrolled inner loop, 16 reads from each tile',
			'The loads that fill `As` and `Bs` each phase',
			'One per phase of a 512 x 512 multiply',
		],
		answer: 1,
		why: 'The inner loop runs `TILE = 16` times and reads one value from `As` and one from `Bs` each time. The tiles are filled with `st.shared`, one per tile.',
	},
	{
		q: 'The Triton TTGIR allocates two 32 x 32 buffers for A and two for B. Why two per matrix?',
		options: [
			'One buffer holds the tile and the other holds its transpose for `tl.dot`',
			'One buffer per warp of the program',
			'The kernel computes two output tiles per program',
			'The copy for the next iteration runs while the other buffer is used',
		],
		answer: 3,
		why: 'This is pipelining. With `num_stages = 3`, Triton keeps `num_stages - 1 = 2` buffers per operand.',
	},
	{
		q: 'In Nsight Compute, the gap in bytes moved from device memory between the naive and tiled kernels is smaller than the gap in global load instructions. Why?',
		options: [
			'The tiled kernel issues more bytes per load instruction',
			'The caches serve part of the naive kernel repeated loads',
			'The profiler counts shared memory loads as global loads',
			'The naive kernel coalesces its loads and the tiled one does not',
		],
		answer: 1,
		why: 'The naive kernel asks for each element many times, and L1 and L2 catch some of those requests before they reach device memory.',
	},
	{
		q: 'What does the tile loader simulation count?',
		options: [
			'Element loads the kernel issues, with no caches or timing',
			'Bytes moved between L2 and device memory over the whole run',
			'Memory transactions after coalescing',
			'Cycles spent waiting at each barrier',
		],
		answer: 0,
		why: 'It counts element loads only. It leaves out L1 and L2, coalescing, bank conflicts and time, and runs the block in lockstep.',
	},
] satisfies Question[];
