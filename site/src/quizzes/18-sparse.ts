import type { Question } from './types';

export default [
	{
		q: 'How much memory does a dense one million by one million matrix of 4 byte floats take?',
		options: ['4 TB', '4 GB', '1 TB', '400 MB'],
		answer: 0,
		why: '10^6 x 10^6 = 10^12 entries at 4 bytes is 4 x 10^12 bytes. 1 TB forgets the 4 bytes per float; a few hundred MB is the sparse form.',
	},
	{
		q: 'What three arrays does COO store?',
		options: [
			'`row_ptr`, `col` and `val`',
			'`col` and `val`, padded to the longest row',
			'The row, the column and the value of each nonzero',
			'The value of each nonzero and its distance to the next one',
		],
		answer: 2,
		why: 'COO keeps one row index, one column index and one value per nonzero, 3 x nnz entries. CSR replaces the row array with `row_ptr`.',
	},
	{
		q: 'For the 4 x 4 example, `row_ptr = [0 2 4 7 9]`. How many nonzeros does row 2 have, and where do they start in `col` and `val`?',
		options: ['2, starting at index 2', '3, starting at index 4', '3, starting at index 2', '7, starting at index 4'],
		answer: 1,
		why: 'Row 2 is entries `row_ptr[2]` to `row_ptr[3] - 1`, 4 to 6, so its length is 7 - 4 = 3. 7 is the end pointer, not a length.',
	},
	{
		q: 'ELL arrays are stored column-major, slot `k` of row `r` at `k * rows + r`. Why?',
		options: [
			'It makes the padding slots take no memory in the arrays',
			'It lets one warp read one whole row in a single transaction',
			'It is the layout cuSPARSE requires',
			'At step `k`, neighbouring threads read neighbouring entries',
		],
		answer: 3,
		why: 'Each thread owns a row and iterates over its slots. With column-major storage, slot `k` of consecutive rows is contiguous, so a warp\'s reads coalesce.',
	},
	{
		q: 'A matrix has 1,000 rows and 5,000 nonzeros. How many array entries does CSR need?',
		options: ['15,000', '10,000', '11,000', '11,001'],
		answer: 3,
		why: 'CSR stores `col` and `val` of length nnz plus `row_ptr` of length rows + 1: 2 x 5,000 + 1,001 = 11,001. 15,000 is COO.',
	},
	{
		q: 'When is ELL the most compact of the three formats?',
		options: [
			'When one row is much longer than the others',
			'When all rows have about the same length',
			'When the matrix has many empty rows',
			'When nnz is larger than the number of columns',
		],
		answer: 1,
		why: 'ELL needs no row array, so with even rows its padding is small and it beats CSR. One long row makes almost the whole array padding.',
	},
	{
		q: 'Why is SpMV limited by memory rather than compute?',
		options: [
			'Each nonzero gives 2 flops for at least 8 bytes read',
			'GPUs cannot do multiply-add on sparse data',
			'The kernel needs an atomic add on `y` for every nonzero',
			'The vector `x` does not fit in shared memory',
		],
		answer: 0,
		why: 'Each nonzero is used once, one multiply and one add, against 4 bytes of column index and 4 of value. That is far below what any GPU can compute per byte it reads.',
	},
	{
		q: 'A warp runs CSR scalar on 32 rows. When is the warp done?',
		options: [
			'When the average row is done',
			'When the first thread finishes its row',
			'When the longest of its 32 rows is done',
			'After a fixed `width` iterations',
		],
		answer: 2,
		why: 'The 32 threads run the loop together, and threads with short rows are masked off and wait. Fixed `width` iterations is the ELL kernel.',
	},
	{
		q: 'Why are the CSR scalar kernel\'s reads of `val` scattered?',
		options: [
			'`val` is stored column-major',
			'Each thread reads `val` indirectly through `x[col[i]]`, a gather',
			'At iteration `j`, neighbouring threads read entries a whole row apart',
			'The kernel reads `val` backwards',
		],
		answer: 2,
		why: 'Thread `r` reads `val[row_ptr[r] + j]`, so for rows longer than a few entries each thread\'s read lands in its own sector.',
	},
	{
		q: 'In the CSR vector kernel, how many `__shfl_down_sync` steps add up the 32 partial sums, and which lane holds the row total?',
		options: ['Five steps, lane 0', '32 steps, lane 31', 'Five steps, lane 31', 'One step, every lane'],
		answer: 0,
		why: 'Offsets 16, 8, 4, 2 and 1 halve the active sums each time, and the kernel writes `y[row]` from lane 0. Five is `log2 32`.',
	},
	{
		q: 'In the CSR vector kernel, how many lanes do nothing in the loop for a row with 3 nonzeros?',
		options: ['0', '3', '16', '29'],
		answer: 3,
		why: 'The row still occupies all 32 lanes, and only lanes 0 to 2 have an entry. Bell and Garland say the vector kernel needs rows longer than the warp size.',
	},
	{
		q: 'Why is `if (row >= rows) return;` safe in the CSR vector kernel?',
		options: [
			'`__shfl_down_sync` ignores exited lanes',
			'`row` is the same for all 32 lanes, so whole warps return together',
			'The kernel is always launched with exactly `rows` warps',
			'Returned lanes contribute zero to the shuffle',
		],
		answer: 1,
		why: 'Every lane that reaches the shuffles has all 31 others with it. Returning per lane breaks this, as exercise 3 shows.',
	},
	{
		q: 'What do the padded slots of ELL cost the kernel?',
		options: [
			'Nothing, since they are skipped',
			'They are loaded and multiplied like real entries',
			'They force an atomic on `y`',
			'They make neighbouring threads diverge',
		],
		answer: 1,
		why: 'Padding is column 0 and value 0, so it adds nothing to the sum, but every thread still runs `width` iterations and reads every slot.',
	},
	{
		q: 'What does the HYB format store?',
		options: [
			'CSR for the short rows and ELL for the few long rows',
			'COO sorted by column',
			'Dense blocks for each nonzero tile, plus CSR for the rest',
			'ELL for the first `K` entries per row, COO for the rest',
		],
		answer: 3,
		why: 'Bell and Garland set `K` to a typical row length, so ELL has little padding and the few long rows spill their extra entries into COO.',
	},
	{
		q: 'What does a COO SpMV kernel with one thread per nonzero need to write `y`?',
		options: [
			'Atomic adds or a segmented reduction across threads',
			'A warp shuffle per row',
			'Nothing extra, since each thread owns its own `y[r]`',
			'A sort of `y` after the kernel',
		],
		answer: 0,
		why: 'The work is perfectly balanced, but nonzeros of one row are on different threads. Their sums must be combined with atomics or a sum that restarts at each row boundary.',
	},
	{
		q: 'In the simulation\'s 12 x 12 power-law matrix, row 0 has 12 nonzeros and the matrix has 41. How many of the scalar kernel\'s thread slots do useful work?',
		options: ['41 of 41', '12 of 144', '41 of 144', '41 of 384'],
		answer: 2,
		why: 'The warp runs 12 iterations for 12 rows, 144 slots, and 41 do work, about 28%. 384 is the vector kernel\'s 12 rows of 32 lanes.',
	},
	{
		q: 'On the same power-law matrix, how many entries does ELL need for `col` and `val` together?',
		options: ['288', '95', '123', '144'],
		answer: 0,
		why: 'ELL stores 2 x 12 rows x width 12 = 288 entries. CSR needs 2 x 41 + 13 = 95 and COO 3 x 41 = 123.',
	},
	{
		q: 'In the simulation, how many steps does the CSR vector kernel take on any of its matrices?',
		options: ['One per nonzero of the longest row', '12', '5', '6'],
		answer: 3,
		why: 'No row is longer than 32, so every warp makes one pass and then 5 shuffle steps: 6. The model counts a shuffle step like a memory step, which real hardware does not.',
	},
	{
		q: 'A warp runs CSR scalar on 32 rows: one with 64 nonzeros, 31 with 2. What fraction of thread slots do useful work?',
		options: ['About 50%', 'About 6%', 'About 3%', 'About 19%'],
		answer: 1,
		why: 'The warp runs 64 iterations: 32 x 64 = 2,048 slots, for 64 + 31 x 2 = 126 multiplies, about 6%.',
	},
	{
		q: 'In Nsight Compute, what should the sectors per request of global loads look like?',
		options: [
			'Equal for all three kernels',
			'High for `spmv_ell`, low for `spmv_csr_scalar`',
			'High for `spmv_csr_scalar`, lower for the other two',
			'High for `spmv_csr_vector`, because of its warp shuffles',
		],
		answer: 2,
		why: 'The scalar kernel\'s threads read entries a row apart, so each request touches many sectors. ELL and CSR vector have neighbouring lanes read neighbouring entries.',
	},
] satisfies Question[];
