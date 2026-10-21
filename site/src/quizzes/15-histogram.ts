import type { Question } from './types';

export default [
	{
		q: 'Why can `hist[b]++` lose counts when many GPU threads run it?',
		options: [
			'The compiler drops increments it proves redundant',
			'It is a read, an add and a write, and two threads can both read before either writes',
			'Global memory only accepts one write per warp per cycle',
			'Integer counters overflow when many threads add at once',
		],
		answer: 1,
		why: 'Two threads that read the same old value both write old + 1, so one count vanishes. Nothing overflows or gets optimized away; the race is in the three separate steps.',
	},
	{
		q: 'At what scope is a plain `atomicAdd` atomic?',
		options: [
			'Within one warp',
			'Within one block',
			'Across the CPU and other GPUs',
			'Across all blocks of the grid on the device',
		],
		answer: 3,
		why: 'Plain `atomicAdd` has device scope, which is why the global kernel is correct across blocks. `atomicAdd_block` is the block scoped one, and `atomicAdd_system` also covers the CPU and other GPUs.',
	},
	{
		q: 'CUDA atomics such as `atomicAdd` have ordering `cuda::memory_order_relaxed`. What does that mean for a histogram?',
		options: [
			'Counts are exact, but other memory accesses are not ordered around the atomic, which a histogram does not need',
			'Counts can be lost unless a `__threadfence()` follows each atomic',
			'The atomic only works on shared memory, not global memory',
			'Each atomic waits for all earlier writes of the grid to finish',
		],
		answer: 0,
		why: 'Relaxed ordering still makes the read, add and write one indivisible step, so no count is lost. It only means the atomic does not order other loads and stores, and a histogram does not rely on that.',
	},
	{
		q: 'Per the Programming Guide, what happens when several threads of a warp do an atomic on the same global address?',
		options: [
			'Only one of the atomics takes effect',
			'The hardware adds them in a tree in one step',
			'Every update happens, one after another, in an undefined order',
			'The warp diverges and replays the instruction until it succeeds',
		],
		answer: 2,
		why: 'The guide says each read, modify and write occurs and they are all serialized, in an undefined order. No update is lost, but they queue, which is the root of contention.',
	},
	{
		q: 'Two inputs of the same size go through the global atomic kernel. One spreads values over all bins, the other puts every value in one bin. Which statement fits the chapter?',
		options: [
			'Both take the same time, since the element count is the same',
			'The one bin input is faster, because one counter stays in cache',
			'The spread input is slower, because it touches more addresses',
			'The one bin input is slower, because every thread queues on one counter',
		],
		answer: 3,
		why: 'Atomics on different addresses run at the same time, while atomics on one address queue. The cost is set by the busiest bin, so the time depends on the data, not only its size.',
	},
	{
		q: 'A kernel sums `float` values into one counter with `atomicAdd`. Why can the result differ in its last bits from run to run?',
		options: [
			'Floating point addition is not associative, and the order of the queued atomics is undefined',
			'Float atomics are not atomic, so some additions are lost',
			'The GPU rounds each atomic result to half precision',
			'Shared memory stores floats with fewer bits than global memory',
		],
		answer: 0,
		why: 'The atomics are all applied, but in an undefined order, and float addition in a different order can round differently. Integer addition gives the same total in any order, so an integer histogram is exact.',
	},
	{
		q: 'Which order of steps matches the chapter\'s privatized kernel?',
		options: [
			'Shared atomics into the copy, zero the copy, `__syncthreads()`, merge',
			'Zero the copy, `__syncthreads()`, shared atomics into the copy, `__syncthreads()`, merge nonzero bins with global atomics',
			'Global atomics, `__syncthreads()`, copy the global histogram into shared memory',
			'Zero the copy, shared atomics into the copy, merge, `__syncthreads()`',
		],
		answer: 1,
		why: 'The first barrier makes sure every bin is zero before anyone adds, and the second makes every count final before the merge reads it. Moving either barrier breaks the kernel.',
	},
	{
		q: 'How much shared memory does one block\'s private copy of a 256 bin histogram of 4 byte counters take?',
		options: ['256 bytes', '4 KB', '1 KB', '32 bytes'],
		answer: 2,
		why: 'The copy is `BINS * 4` bytes, 256 x 4 = 1,024 bytes. 256 bytes forgets the 4 bytes per counter.',
	},
	{
		q: 'After privatization, at most how many global atomics does the global histogram receive?',
		options: [
			'One per bin per block',
			'One per element',
			'One per block',
			'One per bin for the whole grid',
		],
		answer: 0,
		why: 'Each block adds each of its nonzero bins once in the merge. That replaces one global atomic per element with at most one per bin per block.',
	},
	{
		q: 'Why did the shared memory histogram win on Maxwell but usually lose on Kepler in Sakharnykh\'s tests?',
		options: [
			'Kepler had no shared memory',
			'Maxwell had a larger L2 cache that held the whole histogram',
			'Kepler serialized global atomics but not shared ones',
			'Kepler emulates shared memory atomics in software, while Maxwell supports them in hardware',
		],
		answer: 3,
		why: 'On Kepler the shared atomics were slow and its global atomics got help from conflict resolution in the L2 cache. A private copy only pays off when updating it is cheap.',
	},
	{
		q: 'In the simulation, 32 threads count 128 values in 4 rounds. With the one hot bin input, how many slots does the global kernel take?',
		options: ['32', '4', '36', '128'],
		answer: 3,
		why: 'All 32 threads queue on one counter in each of the 4 rounds: 4 x 32 = 128 slots. 32 counts only one round.',
	},
	{
		q: 'Same hot bin input, shared kernel with 4 blocks of 8 threads. How many slots?',
		options: ['32', '36', '128', '40'],
		answer: 1,
		why: 'Each round, 8 threads queue on the hot bin of their copy: 4 x 8 = 32 slots. The merge then queues 4 blocks on one global bin, 4 more slots, so 36.',
	},
	{
		q: 'With the hot bin input and a single block, the shared kernel takes 129 slots against 128 for the global kernel. Why?',
		options: [
			'The private copy is just a second histogram with all 32 threads on it, plus one merge atomic',
			'Shared atomics cost one slot more than global ones in the model',
			'The barrier costs one slot',
			'One thread is left idle in each round',
		],
		answer: 0,
		why: 'With one block every thread queues on the copy, exactly as on the global histogram, and the merge adds one atomic. The model gives shared and global atomics the same cost; privatization needs many copies to help.',
	},
	{
		q: 'For the hot bin input with `B` blocks, the model\'s shared kernel takes `128 / B + B` slots. Which block count among 1, 2, 4, 8, 16 and 32 gives the fewest?',
		options: ['32, because more copies always help', '4', '8 or 16, at 24 slots', '1, because it has only one merge atomic'],
		answer: 2,
		why: '128 / B + B gives 129, 66, 36, 24, 24 and 36. Fewer blocks leave long queues in the copies; more blocks make the merge queue long.',
	},
	{
		q: 'On the uniform input the shared kernel takes 16 slots with 4 blocks but 18 with 8. What causes the increase?',
		options: [
			'Each round has longer queues with 8 blocks',
			'With 8 blocks almost every bin is nonzero in every block, so the merge queues up to 8 atomics on one bin',
			'The model charges extra for every block launch',
			'Uniform data has fewer distinct values with 8 blocks',
		],
		answer: 1,
		why: 'Smaller blocks shorten the queues in each copy, but every copy adds one merge atomic per nonzero bin. With 8 blocks the merge grows to 61 global atomics and up to 8 queue on one bin.',
	},
	{
		q: 'Someone deletes the second `__syncthreads()` in `hist_shared`, the one before the merge, and their test still passes. What is the problem?',
		options: [
			'Nothing, because shared memory atomics already order the merge',
			'The kernel deadlocks on larger inputs',
			'The global histogram counts some elements twice',
			'A thread can merge `local[b]` before other threads finish adding to it, so counts go missing depending on scheduling',
		],
		answer: 3,
		why: 'Without the barrier the merge can read a bin that is still growing, and the later additions are never merged. Whether it happens depends on warp scheduling, so a test can pass by luck; `compute-sanitizer --tool racecheck` reports it.',
	},
	{
		q: 'In the model\'s skewed input with 4 blocks and 8 bins, the merge sends 20 global atomics. How many would it send without the `local[b] > 0` check?',
		options: ['20', '32', '8', '128'],
		answer: 1,
		why: 'Without the check every block adds every bin: 4 x 8 = 32. 128 is the number of elements, the count for the global kernel.',
	},
	{
		q: 'The Triton kernels pass `sem="relaxed"` to `tl.atomic_add`. Why?',
		options: [
			'Relaxed is required for `tl.histogram` to compile',
			'It is the only ordering Triton supports',
			'It matches the ordering of CUDA\'s `atomicAdd`, while Triton\'s default is `acq_rel`',
			'It makes the atomics run in shared memory',
		],
		answer: 2,
		why: 'A histogram only needs atomic counts, which is what CUDA\'s relaxed `atomicAdd` gives. Triton defaults to `acq_rel`, a stronger ordering the histogram does not need.',
	},
] satisfies Question[];
