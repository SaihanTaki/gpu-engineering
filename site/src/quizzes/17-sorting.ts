import type { Question } from './types';

export default [
	{
		q: 'Why does quicksort fit a GPU poorly, according to the chapter?',
		options: [
			'It needs more memory than the GPU has',
			'It only sorts integers',
			'It cannot be written without recursion',
			'It decides what to compare next from earlier results, so lanes of a warp branch on their own data',
		],
		answer: 3,
		why: 'Data-dependent control flow makes a warp run both paths. Sorting networks fix every comparison in advance, and radix sort does not compare keys at all.',
	},
	{
		q: 'In a sorting network, what is a stage?',
		options: [
			'One pass of a single comparator along the whole array',
			'A set of comparators in which every wire is in at most one comparator, so all can run at once',
			'One sorted run of length `k`',
			'One kernel launch that sorts a tile in shared memory',
		],
		answer: 1,
		why: 'Comparators on different wires are independent, so a stage runs them in parallel, one thread each, with a barrier at the end.',
	},
	{
		q: 'For a bitonic network on random, sorted and reversed input of the same size, what changes?',
		options: [
			'Only which pairs swap',
			'The number of stages',
			'The memory addresses read',
			'The number of comparators',
		],
		answer: 0,
		why: 'The comparisons do not depend on the keys, so the work, addresses and stage count are fixed. Whether a pair swaps is the only data-dependent step, and it compiles to a select.',
	},
	{
		q: 'Which sequence is bitonic?',
		options: ['1 4 2 8 3 7', '8 1 7 2 6 3', '1 4 6 8 7 5 3 2', '5 1 5 1 5 1'],
		answer: 2,
		why: 'A bitonic sequence first rises and then falls, or is a rotation of such a sequence. The others rise and fall more than once.',
	},
	{
		q: 'How many stages does it take to merge one bitonic sequence of length `L` into sorted order?',
		options: ['`log2 L`', '`L / 2`', '`L`', '`(log2 L)²`'],
		answer: 0,
		why: 'Each stage halves the comparator distance, from `L/2` down to 1, so there are `log2 L` stages. `L / 2` is the number of comparators per stage.',
	},
	{
		q: 'How many stages and comparators does the bitonic network need for 16 keys?',
		options: ['6 stages, 24 comparators', '4 stages, 32 comparators', '16 stages, 128 comparators', '10 stages, 80 comparators'],
		answer: 3,
		why: 'With `p = 4` there are 1 + 2 + 3 + 4 = 10 stages of 16 / 2 = 8 comparators: 80. 6 and 24 are the counts for 8 keys.',
	},
	{
		q: 'In the bitonic loop, what do `k` and `j` stand for?',
		options: [
			'`k` is the thread index, `j` the key index',
			'`k` is the number of keys, `j` the number of stages',
			'`k` is the length of the sorted runs being built, `j` the comparator distance',
			'`k` is the comparator distance, `j` the run length',
		],
		answer: 2,
		why: 'The outer loop doubles `k` from 2 to `n`, and the inner loop halves `j` from `k / 2` to 1, comparing `i` with `i + j`.',
	},
	{
		q: 'In the simulation, 8 already sorted keys go through the network. How many swaps happen?',
		options: ['0', '10', '14', '24'],
		answer: 1,
		why: 'The early phases sort half the runs downward, so sorted pairs swap, and the last phase undoes that: 10 swaps. 24 is the comparator count, and 14 is the reversed input.',
	},
	{
		q: 'Thread `t` runs comparator `t` with `i = (t / j) * 2 * j + t % j`. For `j = 2`, which key does thread 3 compare first?',
		options: ['3', '5', '6', '7'],
		answer: 1,
		why: '`(3 / 2) * 4 + 3 % 2 = 4 + 1 = 5`, compared with `5 + 2 = 7`. The formula skips the upper half of every group of `2j`, so no thread idles.',
	},
	{
		q: 'Why does the chapter prefer `n / 2` threads with that index formula over one thread per key?',
		options: [
			'It needs no barrier between stages',
			'It avoids bank conflicts',
			'It works for any `n`, not only powers of two',
			'One thread per key idles half the threads; the formula keeps every thread busy',
		],
		answer: 3,
		why: 'With one thread per key, only threads with `(i ^ j) > i` work. The formula gives each of the `n / 2` threads one comparator.',
	},
	{
		q: 'In `sort_global`, what acts as the barrier between stages?',
		options: [
			'The kernel launch boundary',
			'`__syncthreads()`',
			'A `__threadfence()` after each swap',
			'An atomic counter',
		],
		answer: 0,
		why: 'The host runs one launch per stage, and no thread of the next launch starts before the previous one has finished. `__syncthreads()` only syncs one block, which is why it serves the shared memory version.',
	},
	{
		q: 'For `2^20` keys, how many launches does `sort_global` make?',
		options: ['20', '55', '210', '400'],
		answer: 2,
		why: 'One per stage: 20 x 21 / 2 = 210. 55 is the number of global stages left in the shared memory version.',
	},
	{
		q: 'For `2^20` keys with 1024-key tiles, how many passes over the array does `sort_shared` make?',
		options: ['210', '11', '55', '66'],
		answer: 3,
		why: '55 global stages (1 + 2 + ... + 10, for `j >= 1024`) plus 11 shared launches, one for phases 2 to 1024 and one for each later phase: 66.',
	},
	{
		q: 'In `bitonic_shared`, stages with `j < 32` read lower keys spread over 64 consecutive words. What does that cause?',
		options: [
			'A 2-way bank conflict',
			'Uncoalesced global loads',
			'A 32-way bank conflict',
			'No conflict, because the words are consecutive',
		],
		answer: 0,
		why: '64 consecutive words map two per bank, so each bank serves two lanes. For `j >= 32` the lanes read 32 consecutive words, conflict free.',
	},
	{
		q: 'The kernels pad the input to a power of two with `+inf`. How many keys does an input of 1025 keys sort as?',
		options: ['1025', '2048', '1024', '4096'],
		answer: 1,
		why: 'The next power of two above 1025 is 2048, which almost doubles the work. The `+inf` keys sort to the end and are dropped.',
	},
	{
		q: 'Why do NaN keys break the padded bitonic sort?',
		options: [
			'NaN cannot be stored in shared memory',
			'NaN sorts before `+inf`',
			'Every comparison with NaN is false',
			'NaN turns the network data dependent',
		],
		answer: 2,
		why: 'A compare-swap with a NaN key always sees the pair as in order, so the network cannot order those keys. Storage and data dependence are not the issue; the comparison is.',
	},
	{
		q: 'An LSD radix sort pass must be stable. What does that mean here?',
		options: [
			'It always produces the same output for the same input',
			'It never runs out of shared memory',
			'It processes the most significant digit first',
			'Keys with the same digit keep their order from the previous pass',
		],
		answer: 3,
		why: 'Each pass sorts only by one digit, lowest first, so stability is what keeps the order set by earlier digits. Computing ranks in block order keeps it.',
	},
	{
		q: 'Which three patterns make up one radix sort pass, in order?',
		options: [
			'Histogram, exclusive scan, scatter',
			'Scan, histogram, reduction',
			'Compare-swap, merge, scatter',
			'Reduction, scatter, histogram',
		],
		answer: 0,
		why: 'Count digits per block, scan the counts laid out digit by digit to get output offsets, then scatter each key to its offset plus its rank.',
	},
	{
		q: 'A radix sort of 32-bit keys uses 8-bit digits. How many passes, and how many bins per pass?',
		options: ['8 passes of 16 bins', '3 passes of 2048 bins', '32 passes of 2 bins', '4 passes of 256 bins'],
		answer: 3,
		why: '32 / 8 = 4 passes, each counting `2^8 = 256` digit values. Wider digits mean fewer passes but more bins to count and scan.',
	},
	{
		q: 'What does merge path find with its binary search?',
		options: [
			'The largest key in each run',
			'How many of the first `d` output keys come from `A` and how many from `B`',
			'The position of each key in the final sorted array by comparing with all others',
			'The run length for the next merge pass',
		],
		answer: 1,
		why: 'Output position `d` takes `x` keys from `A` and `d - x` from `B`. Each thread searches its two endpoints and then merges its own slices sequentially. The splits depend on the data, but the work per thread does not.',
	},
] satisfies Question[];
