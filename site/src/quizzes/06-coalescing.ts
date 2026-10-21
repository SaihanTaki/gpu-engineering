import type { Question } from './types';

export default [
	{
		q: 'What size are the transactions that serve global memory, according to the CUDA Programming Guide?',
		options: [
			'4 bytes, one per float',
			'128 bytes, a full cache line',
			'32 bytes, aligned to a 32-byte boundary',
			'32 bytes, starting at any address',
		],
		answer: 2,
		why: 'Transactions are 32 bytes and aligned: bytes 0 to 31 or 32 to 63, never 20 to 51. Nsight Compute calls such a piece a sector.',
	},
	{
		q: 'How does Nsight Compute relate sectors to cache lines?',
		options: ['A line is four sectors, 128 bytes', 'A line is two sectors, 64 bytes', 'A line is eight sectors, 256 bytes', 'A line is 32 sectors, one per lane'],
		answer: 0,
		why: 'An L1 or L2 cache line is four 32-byte sectors. Lines matter for the caches; the transaction count is in sectors.',
	},
	{
		q: 'On compute capability 6.0 and later, what decides the cost of one warp\'s load?',
		options: [
			'The number of active lanes',
			'The number of distinct cache lines its addresses touch',
			'The number of bytes the lanes ask for',
			'The number of distinct 32-byte sectors its addresses fall into',
		],
		answer: 3,
		why: 'The accesses coalesce into as many 32-byte transactions as it takes to serve all lanes. Two loads can ask for the same bytes and touch very different numbers of sectors.',
	},
	{
		q: 'A warp loads one 4-byte float per lane, and every lane falls in a different sector. How many sectors and bytes does the warp fetch?',
		options: ['8 sectors, 256 bytes', '32 sectors, 1,024 bytes', '32 sectors, 4,096 bytes', '128 sectors, 4,096 bytes'],
		answer: 1,
		why: '32 sectors of 32 bytes is 1,024 bytes for 128 bytes used. 4,096 bytes would be 32 whole cache lines, but the unit is the sector.',
	},
	{
		q: 'A warp reads floats with `in[i + 1]` from memory returned by `cudaMalloc`. How many sectors does it touch, and how much of the fetched data is used?',
		options: ['4 sectors, 100%', '8 sectors, 50%', '32 sectors, 12.5%', '5 sectors, 80%'],
		answer: 3,
		why: 'The warp reads bytes 4 to 131, so sectors 0 to 4: 160 bytes fetched for 128 used. A misaligned copy is only a little worse than an aligned one.',
	},
	{
		q: 'A warp reads floats with `in[i + offset]`. Which offset keeps the count at 4 sectors?',
		options: ['1', '4', '8', '12'],
		answer: 2,
		why: 'Only offsets that are a multiple of 8 floats (32 bytes) start the warp on a sector boundary. An offset of 4 floats is 16 bytes, which still needs a fifth sector.',
	},
	{
		q: 'A warp reads floats with `in[i * 2]`. How many sectors does it touch, and how much of the fetched data is used?',
		options: ['8 sectors, 50%', '4 sectors, 100%', '16 sectors, 25%', '6 sectors, 66.7%'],
		answer: 0,
		why: 'Lanes are 8 bytes apart, so the warp spans 256 bytes in 8 sectors and uses half of each. Stride 4 would be 16 sectors and 25%.',
	},
	{
		q: 'How does a float load with stride 16 compare with one at stride 8?',
		options: [
			'Stride 16 needs 64 sectors',
			'Both need 32 sectors; stride 16 only spreads them over more lines',
			'Stride 16 needs 16 sectors',
			'Stride 8 needs 16 sectors and stride 16 needs 32',
		],
		answer: 1,
		why: 'From stride 8 on, lanes are at least 32 bytes apart and each needs its own sector. A warp has only 32 lanes, so it cannot need more than 32 sectors.',
	},
	{
		q: 'A full warp loads 16-byte `float4` values with `in[i]`. How many sectors does it touch, and how much of the fetched data is used?',
		options: ['4 sectors, 100%', '16 sectors, 25%', '16 sectors, 100%', '32 sectors, 50%'],
		answer: 2,
		why: 'The warp asks for 512 bytes, which is 16 sectors in 4 lines, all used. More sectors than the float case does not mean waste when every byte is used.',
	},
	{
		q: 'A kernel reads a whole `struct { float x, y, z; }` per thread. What does the 12.4 Programming Guide imply for it?',
		options: [
			'It compiles to one 12-byte load instruction',
			'It always compiles to one 16-byte load instruction',
			'Structs cannot be read from global memory',
			'It takes several instructions, and `__align__(16)` pads it so one 16-byte load can fetch it',
		],
		answer: 3,
		why: 'A single global memory instruction needs a size of 1, 2, 4, 8 or 16 bytes, naturally aligned. 12 bytes is neither, so padding to 16 bytes is the fix.',
	},
	{
		q: 'Particles are stored as an array of `struct { float x, y, z; }`. A warp reads only `p[i].x`. How many sectors does it touch?',
		options: ['12 sectors, 33.3% used', '4 sectors, 100% used', '3 sectors, 100% used', '32 sectors, 12.5% used'],
		answer: 0,
		why: 'Reading one field is a stride-3 load: lanes 12 bytes apart span 384 bytes in 12 sectors, for 128 bytes used. Stride 3 is less than one sector per lane, so it is not the 32-sector worst case.',
	},
	{
		q: 'The particle struct grows from 3 floats to 6, and the kernel still reads only `x`. What happens to the sectors one warp touches?',
		options: [
			'With AoS it grows to 24 sectors; with SoA it stays at 4',
			'Both layouts grow to 24 sectors',
			'With AoS it stays at 12 sectors; with SoA it grows to 8',
			'Neither layout changes',
		],
		answer: 0,
		why: 'With AoS the stride grows with the struct, here 24 bytes per lane. With SoA, `x[i]` stays a contiguous load whatever the number of fields.',
	},
	{
		q: 'In which case is an array of structs no worse than a struct of arrays?',
		options: [
			'A kernel reads one field of a 12-byte struct',
			'A kernel reads every field of an aligned 16-byte struct',
			'A kernel reads one field out of eight',
			'A kernel updates positions and never reads masses',
		],
		answer: 1,
		why: 'When every field is read, every fetched byte is used, and an aligned 16-byte struct loads in one instruction per lane. SoA wins when a kernel reads some fields and not others.',
	},
	{
		q: 'The chapter\'s kernels write `out[i]` contiguously on purpose. What would a strided store cost?',
		options: [
			'Nothing extra, because stores are buffered',
			'Half the cost of a strided load',
			'Nothing extra, because stores always coalesce',
			'The same as a strided load, because stores follow the same 32-byte rule',
		],
		answer: 3,
		why: 'Stores are served by 32-byte sectors too. The kernels keep the writes contiguous so that only the loads change.',
	},
	{
		q: 'Which Nsight Compute number corresponds to the simulation\'s sector count per warp?',
		options: [
			'Achieved Occupancy in the Occupancy section',
			'Branch efficiency in Source Counters',
			'Sectors/Req for global loads in the L1/TEX cache table',
			'Stall Long Scoreboard in Warp State Statistics',
		],
		answer: 2,
		why: 'Sectors/Req in the Memory Workload Analysis tables is the number of sectors per load request. Expect 4 for the aligned copy and 32 from stride 8 on.',
	},
	{
		q: 'A strided kernel over a small array causes less DRAM traffic than its sector count suggests. Why?',
		options: [
			'The sector count overestimates how many sectors a float load touches',
			'A sector one warp fetched and did not use may still be in L1 or L2 when a neighbouring warp asks for it',
			'DRAM compresses unused bytes before sending them',
			'Small arrays are always aligned to 256 bytes',
		],
		answer: 1,
		why: 'The simulation counts the sectors one load instruction touches and leaves out the caches. Reuse across warps can make the real DRAM cost lower.',
	},
	{
		q: 'A row-major float matrix has 512 columns. One warp reads column `c`, one row per lane. How many sectors does the load touch?',
		options: ['4 sectors, 100% used', '16 sectors, 25% used', '8 sectors, 50% used', '32 sectors, 12.5% used'],
		answer: 3,
		why: 'Neighbouring lanes are one row, 2 KB, apart, far more than a sector. Reading a column of a row-major matrix is a load with a stride of the row length.',
	},
	{
		q: 'Why does a warp reading `in[i]` with no offset start exactly at the beginning of a sector?',
		options: [
			'`cudaMalloc` returns memory aligned to at least 256 bytes',
			'The compiler pads every array to a multiple of 32 bytes',
			'The L1 cache realigns each load',
			'Lane 0 always reads from address 0',
		],
		answer: 0,
		why: 'An allocation aligned to 256 bytes also starts on a 32-byte boundary. An array that starts partway into a buffer loses that alignment.',
	},
	{
		q: 'In Triton, `tl.load(src_ptr + offs * stride)` runs with `stride` 2. What does the load cost?',
		options: [
			'The same as a contiguous load, because Triton reorders the threads',
			'Nothing, because Triton rejects strides other than 1',
			'Still twice the sectors of a contiguous load',
			'The same as a contiguous load, because Triton converts the data to SoA',
		],
		answer: 2,
		why: 'Triton picks which thread loads which element, but it cannot change the addresses themselves. The fraction of fetched bytes used depends only on the stride.',
	},
	{
		q: 'Why does the CUDA `copy_strided` compute its index as `size_t`?',
		options: [
			'`size_t` loads are faster than `int` loads',
			'So that `i * stride` cannot overflow `int` for large arrays',
			'Numba requires 64-bit indices',
			'It aligns the load to 8 bytes',
		],
		answer: 1,
		why: 'A large array times a large stride can pass the largest `int`. The index type does not change the size or alignment of the float being loaded.',
	},
] satisfies Question[];
