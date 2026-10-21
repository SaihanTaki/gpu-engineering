import type { Question } from './types';

export default [
	{
		q: 'How do you mark a C++ function as a CUDA kernel?',
		options: ['`__device__`', '`__kernel__`', '`__host__`', '`__global__`'],
		answer: 3,
		why: 'A kernel is a function marked `__global__`. The host launches it and the device runs it once per thread.',
	},
	{
		q: 'In `whoami<<<A, B>>>(...)`, what do `A` and `B` set?',
		options: [
			'`A` is the threads per block, `B` is the number of blocks',
			'`A` is the number of blocks, `B` is the threads per block',
			'`A` is the total number of threads, `B` is the number of SMs',
			'`A` is the array length, `B` is the threads per block',
		],
		answer: 1,
		why: 'The first value is `gridDim`, the number of blocks, and the second is `blockDim`, the threads in each block.',
	},
	{
		q: 'What is the most threads one block can hold?',
		options: ['1024', '256', '2048', '`2^31 - 1`'],
		answer: 0,
		why: 'A block holds at most 1024 threads. 2^31 - 1 is the limit on the number of blocks in one dimension, not threads per block.',
	},
	{
		q: 'What does `threadIdx.x` hold?',
		options: [
			'The thread\'s position in the whole grid',
			'The number of threads in the block',
			'The thread\'s position inside its block',
			'The index of the SM running the thread',
		],
		answer: 2,
		why: '`threadIdx` is the position inside the block. The position in the whole grid needs `blockIdx` and `blockDim` as well.',
	},
	{
		q: 'In a normal launch, why are blocks not allowed to wait for each other?',
		options: [
			'Blocks have no access to global memory while they run',
			'Each block runs on its own separate GPU in the machine',
			'Blocks always run one after another, in the order of `blockIdx`',
			'So the hardware can hand blocks to any free SM, in any order',
		],
		answer: 3,
		why: 'Because blocks are independent, the same kernel runs on a GPU with a few dozen SMs or with more than a hundred. Blocks run in any order, not by `blockIdx`.',
	},
	{
		q: 'A kernel is launched with 64 threads per block. Which `i = blockIdx.x * blockDim.x + threadIdx.x` does thread 10 of block 5 compute?',
		options: ['74', '330', '650', '320'],
		answer: 1,
		why: '`5 * 64 + 10 = 330`. 320 is where block 5 starts, and 650 swaps the roles of block and thread.',
	},
	{
		q: 'For `n = 20` and 8 threads per block, the launch uses `n / threads` blocks. What happens?',
		options: [
			'Elements 16 to 19 are never written, and no error is reported',
			'The launch fails with an error',
			'The last block automatically gets extra threads',
			'Threads 16 to 19 write past the end of the array',
		],
		answer: 0,
		why: '`20 / 8` is 2 in integer arithmetic, so only 16 threads run. Nothing fails, the last four elements are just wrong.',
	},
	{
		q: 'With 128 threads per block and `(n + threads - 1) / threads` blocks, how many blocks run for `n = 1000`?',
		options: ['7', '9', '8', '1000'],
		answer: 2,
		why: '`(1000 + 127) / 128 = 1127 / 128`, which is 8 in integer division. Plain `1000 / 128` gives 7 and skips elements 896 to 999.',
	},
	{
		q: 'In that launch of 8 blocks of 128 threads for `n = 1000`, how many threads fail `if (i < n)`?',
		options: ['0', '128', '104', '24'],
		answer: 3,
		why: '8 * 128 = 1024 threads for 1000 elements, so the 24 threads with `i` from 1000 to 1023 skip the store.',
	},
	{
		q: 'A rounded-up launch has no `if (i < n)` check. What can the extra threads do?',
		options: [
			'Nothing, the hardware drops writes past the end of an array',
			'Write past the end of the array, which is undefined behavior',
			'Always crash the kernel with a clear error',
			'Overwrite the first elements of the same array',
		],
		answer: 1,
		why: 'Out of bounds writes are undefined behavior. They may corrupt another buffer, crash with an illegal address error, or appear to work for a while.',
	},
	{
		q: 'Which block shape is not allowed?',
		options: ['`dim3(32, 32)`', '`dim3(64, 16)`', '`dim3(256, 4)`', '`dim3(64, 32)`'],
		answer: 3,
		why: 'The threads per block is the product of the dimensions. `64 * 32 = 2048`, over the 1024 limit. The other three are exactly 1024.',
	},
	{
		q: 'An image is 100 pixels wide and 50 high. With `dim3 block(16, 16)` and a rounded-up grid, how many blocks run?',
		options: ['28', '18', '24', '35'],
		answer: 0,
		why: '`(100 + 15) / 16 = 7` blocks across and `(50 + 15) / 16 = 4` down, so 7 * 4 = 28. Without rounding up you get 6 * 3 = 18.',
	},
	{
		q: 'In the 2D kernel, a thread has `row` and `col`. Which offset does it write into the row-major array?',
		options: ['`col * height + row`', '`row * height + col`', '`row * width + col`', '`col * width + row`'],
		answer: 2,
		why: 'In row-major order each row holds `width` elements, so the offset is `row * width + col`. Using `height` would only be right for a column-major layout.',
	},
	{
		q: 'Why does the 2D kernel map `x` to columns rather than rows?',
		options: [
			'Consecutive `threadIdx.x` values then touch consecutive addresses',
			'The `x` dimension allows far more blocks than the `y` dimension',
			'Rows must always be handled by `blockIdx` rather than by the threads',
			'It avoids the need for a separate bounds check on the rows',
		],
		answer: 0,
		why: 'Neighboring threads in `x` reading neighboring addresses is the access pattern memory serves fastest. Chapter 6 measures it.',
	},
	{
		q: 'In Numba, what does `cuda.grid(1)` return?',
		options: [
			'The number of blocks in the grid',
			'`blockIdx.x * blockDim.x + threadIdx.x`',
			'`threadIdx.x` only',
			'The total number of threads launched',
		],
		answer: 1,
		why: '`cuda.grid(1)` is a shortcut for the global index. The launch itself is written `whoami[blocks, threads](...)`.',
	},
	{
		q: 'In the Triton version, what plays the role of the `if (i < n)` bounds check?',
		options: ['`tl.program_id(0)`', 'The `num_warps` launch option', 'The offsets from `tl.arange(0, BLOCK)`', 'The mask on the loads and stores'],
		answer: 3,
		why: 'A Triton program handles a whole block of offsets, and the mask turns off the ones past the end. `tl.program_id(0)` plays the role of `blockIdx.x`.',
	},
	{
		q: 'Why must Triton\'s `BLOCK` be a power of two?',
		options: [
			'Triton launches one warp per element',
			'GPU memory only accepts power of two addresses',
			'`tl.arange` only accepts powers of two',
			'The mask cannot handle other sizes',
		],
		answer: 2,
		why: 'The offsets come from `tl.arange(0, BLOCK)`, which only accepts powers of two. The mask handles any array length.',
	},
] satisfies Question[];
