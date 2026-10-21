import type { Question } from './types';

export default [
	{
		q: 'A naive kernel blurs a 4096 x 4096 float image with a 5 x 5 filter, one thread per output, each reading its whole window from global memory. How many loads does it issue in total?',
		options: ['16,777,216', '83,886,080', '419,430,400', '67,108,864'],
		answer: 2,
		why: 'There are 4096 x 4096 = 16,777,216 outputs and each reads 25 pixels: 419,430,400 loads. 16,777,216 is one load per pixel, what an ideal kernel would need.',
	},
	{
		q: 'What makes a convolution a special case of a stencil?',
		options: [
			'It uses a full window with a weight for every position',
			'It only uses the four nearest neighbours of each pixel',
			'It reads each input pixel exactly once',
			'It must flip the filter before applying it to the image',
		],
		answer: 0,
		why: 'A stencil is any fixed pattern around each position; a convolution uses the whole `(2R + 1) x (2R + 1)` window, each position with its own weight. The four neighbour heat update is a stencil but not a full window.',
	},
	{
		q: 'With the chapter\'s boundary rule, how many real pixels does a corner output sum with a 3 x 3 filter?',
		options: ['9', '6', '1', '4'],
		answer: 3,
		why: 'Pixels outside the image count as zero, and only a 2 x 2 part of the corner window lies inside the image. 6 is the count on an edge, 9 inside.',
	},
	{
		q: 'A block computes `T` consecutive outputs of a 1D convolution with radius `R`. How many input pixels does it need?',
		options: ['`T + R`', '`T + 2R`', '`T x (2R + 1)`', '`2R + 1`'],
		answer: 1,
		why: 'The first output needs `R` pixels before it and the last needs `R` after it, so the block needs `T + 2R` pixels. `T x (2R + 1)` is what the naive kernel loads.',
	},
	{
		q: 'For a 2D tile of `T x T` outputs with radius `R`, what is the halo overhead?',
		options: [
			'`(2R + 1)² / T²`',
			'`((T + 2R)² - T²) / T²`',
			'`(T + 2R)² / T²`',
			'`2R / T`',
		],
		answer: 1,
		why: 'The overhead is halo cells per output cell: the input tile minus the output tile, divided by the output tile. `(T + 2R)² / T²` is the total loads per output, one more than the overhead.',
	},
	{
		q: 'With 16 x 16 tiles and a 5 x 5 filter (R = 2), how many global loads per output does the tiled kernel issue?',
		options: ['25', '1.27', '2.25', '1.56'],
		answer: 3,
		why: 'The input tile is 20 x 20 = 400 pixels for 256 outputs: 400 / 256 = 1.5625. 25 is the naive kernel; 2.25 is the 8 x 8 tile at the same radius.',
	},
	{
		q: 'For 16 x 16 tiles, at which radius does the halo first hold more cells than the output tile itself?',
		options: ['R = 4', 'R = 2', 'R = 3', 'R = 8'],
		answer: 0,
		why: 'At R = 4 the input tile is 24 x 24 = 576 cells, 320 of them halo against 256 outputs, an overhead of 125%. At R = 3 it is 89.1%.',
	},
	{
		q: 'Why does a stencil kernel want the largest tile its block size and shared memory allow?',
		options: [
			'Larger tiles avoid the need for `__syncthreads()` after the load',
			'Larger tiles remove bank conflicts',
			'The halo grows with the border, the outputs with the area',
			'Larger tiles let the filter fit in constant memory',
		],
		answer: 2,
		why: 'The halo is a border of width `R` around the tile, so its share falls as the tile grows. At R = 2 the overhead is 125% for 8 x 8 tiles but 26.6% for 32 x 32.',
	},
	{
		q: 'This chapter\'s tiled kernel uses `T²` threads to fill `(T + 2R)²` shared memory slots. How?',
		options: [
			'It launches `(T + 2R)²` threads and lets the halo threads idle during the compute loop',
			'It reads halo pixels straight from global memory and hopes they are cached',
			'Each thread loads exactly one slot and the halo is left as zero',
			'Thread `t` fills slots `t`, `t + T²`, `t + 2T²` and so on until the tile is full',
		],
		answer: 3,
		why: 'This is the second arrangement: every thread computes, and some load two or more slots. The first and third options are the other two arrangements the chapter lists.',
	},
	{
		q: 'How many load rounds does the tiled kernel need for `T = 16, R = 2`?',
		options: ['1', '2', '3', '25'],
		answer: 1,
		why: 'The rounds are `ceil((T + 2R)² / T²) = ceil(400 / 256) = 2`. In the second round only some threads have a slot left.',
	},
	{
		q: 'In the simulation\'s corner position with `T = 16, R = 2`, the tiled kernel issues 324 global loads instead of 400. Why fewer?',
		options: [
			'Padding slots outside the image get `0.0f` without a load',
			'Corner blocks use a smaller tile that fits inside the image',
			'The cache serves the overlapping halo loads at the corner',
			'Corner threads skip the compute loop',
		],
		answer: 0,
		why: 'Zero padding means the slots outside the image are written as zero without a load. Only the 18 x 18 part of the input tile inside the image is loaded.',
	},
	{
		q: 'How much shared memory does the input tile take for `T = 16, R = 2` with float pixels?',
		options: ['1,024 bytes', '400 bytes', '1,600 bytes', '6,400 bytes'],
		answer: 2,
		why: 'The tile is `4 (T + 2R)²` bytes: 4 x 400 = 1,600. 400 forgets the 4 bytes per float. It is far below the 48 KB a block may declare statically.',
	},
	{
		q: 'Why does `T` stop at 32 for this kernel?',
		options: [
			'Shared memory cannot hold a larger tile',
			'Constant memory limits the tile size',
			'A block holds at most 1,024 threads, one per output',
			'A warp has 32 threads',
		],
		answer: 2,
		why: 'Each output has its own thread, and 32 x 32 = 1,024 is the block limit. Even the largest simulated tile uses only 9,216 bytes of shared memory, so shared memory is not the limit.',
	},
	{
		q: 'Why does the filter suit `__constant__` memory?',
		options: [
			'All threads of a warp read the same weight, which the cache broadcasts',
			'Constant memory is faster than shared memory for any access',
			'Each thread reads a different weight, which constant memory spreads across banks',
			'Constant memory is the only memory that survives between launches',
		],
		answer: 0,
		why: 'In step `(dy, dx)` every thread reads `c_filter[dy * (2R + 1) + dx]`, one address, the case the constant cache broadcasts. Different addresses per thread would serialize the read.',
	},
	{
		q: 'Which call copies the filter into constant memory from the host?',
		options: ['`cudaMemcpy`', '`cudaMalloc`', '`cudaMemcpyFromSymbol`', '`cudaMemcpyToSymbol`'],
		answer: 3,
		why: '`cudaMemcpyToSymbol(c_filter, h_filter, bytes)` writes to the `__constant__` variable. In Numba the equivalent is `cuda.const.array_like`, which fixes the values when the kernel compiles.',
	},
	{
		q: 'The tiled kernel returns early for threads outside the image. Where must that return sit?',
		options: [
			'Before the load loop, so idle threads do no work',
			'After the `__syncthreads()`, so every thread loads and reaches the barrier',
			'Inside the compute loop',
			'Anywhere, because `__syncthreads()` ignores threads that have already exited',
		],
		answer: 1,
		why: 'Threads past the image still have shared memory slots to fill, assigned by thread number, and the barrier must be reached by the whole block. Returning before the load loop leaves slots unfilled and can hang.',
	},
	{
		q: 'In the generated PTX, Numba\'s `conv2d_tiled` has no `ld.const` while CUDA\'s has 25. Why?',
		options: [
			'Numba puts the filter in shared memory',
			'Numba does not support constant memory, so the filter stays in global memory',
			'Numba reads the filter from global memory instead',
			'The values are fixed at compile time and written into the instructions',
		],
		answer: 3,
		why: 'The unrolled loop let NVVM see every weight, so no load was needed. In the naive Numba kernel the outer loop stayed a loop, so its weights are still read with `ld.const`.',
	},
	{
		q: 'How does the Triton kernel handle the boundary rule?',
		options: [
			'It loads a halo into shared memory and zeroes the outside slots',
			'It pads the input tensor with zeros on the host',
			'It masks offsets outside the image in `tl.load` and passes `other=0.0`',
			'It skips programs that touch the edge',
		],
		answer: 2,
		why: 'Masked off offsets read as `other`, so zero padding is one argument. Triton has no hand-loaded halo; the overlapping shifted loads are left to the L1 and L2 caches.',
	},
	{
		q: 'In Nsight Compute, what should the naive kernel\'s L1 hit rate look like, and what follows from it?',
		options: [
			'High, so the measured time gap is smaller than the gap in load counts',
			'Near zero, so the naive kernel is slower than the load counts predict',
			'High, so the tiled kernel is slower than the naive one',
			'Near zero, because constant memory bypasses L1',
		],
		answer: 0,
		why: 'Neighbouring threads load overlapping windows, so many naive loads hit in L1. The simulation leaves the caches out, which is why it shows a larger gap than the GPU will.',
	},
	{
		q: 'A variant stores a per-pixel weight `g[y * w + x]` in `__constant__` memory. What happens to its reads?',
		options: [
			'They are broadcast like the filter reads',
			'The 32 threads read 32 addresses, so the read splits into 32 requests',
			'They become shared memory reads',
			'They are coalesced into one 128 byte transaction per warp, like global reads',
		],
		answer: 1,
		why: 'Constant memory broadcasts only when the warp reads one address. Per-pixel data is also too big for 64 KB of constant memory and belongs in global memory.',
	},
] satisfies Question[];
