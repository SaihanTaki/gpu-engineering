import type { Question } from './types';

export default [
	{
		q: 'What is a CUDA core?',
		options: [
			'A small CPU-like core that runs its own program',
			'A whole streaming multiprocessor',
			'A warp scheduler',
			'One FP32 arithmetic lane inside an SM partition',
		],
		answer: 3,
		why: 'A CUDA core cannot fetch an instruction or run a program by itself. It is one arithmetic lane, driven together with many others by a warp scheduler.',
	},
	{
		q: 'Which order lists the layers of a GPU from the outside in?',
		options: [
			'L2 cache, DRAM, partitions, SMs',
			'DRAM, L2 cache, SMs, partitions',
			'DRAM, SMs, L2 cache, partitions',
			'SMs, partitions, L2 cache, DRAM',
		],
		answer: 1,
		why: 'DRAM sits next to the die, the L2 is one cache on the die, the SMs compute, and each SM is split into partitions.',
	},
	{
		q: 'What does CUDA call the DRAM next to the GPU die?',
		options: ['Shared memory', 'Local cache', 'Global memory', 'Register memory'],
		answer: 2,
		why: 'Everything a kernel reads through a pointer starts in DRAM, which CUDA calls global memory. Shared memory is on chip, inside each SM.',
	},
	{
		q: 'What does all traffic between the SMs and DRAM pass through?',
		options: ['The L2 cache', 'The register file', 'The warp scheduler', 'The CPU'],
		answer: 0,
		why: 'There is one L2 cache on the die, shared by every SM, and all DRAM traffic goes through it.',
	},
	{
		q: 'How many partitions is each SM split into?',
		options: ['2', '32', '4', 'One per CUDA core'],
		answer: 2,
		why: 'Each SM has four partitions, each with its own warp scheduler, slice of the register file and execution units. 32 is the number of threads in a warp.',
	},
	{
		q: 'An A100 has 108 SMs with 64 FP32 cores each. How many FP32 cores does it have?',
		options: ['6,912', '13,824', '1,728', '16,896'],
		answer: 0,
		why: '108 * 64 = 6,912. 16,896 is the H100 figure, 132 SMs with 128 cores each.',
	},
	{
		q: 'When a warp waits for a load from DRAM, what does the warp scheduler do?',
		options: [
			'Stalls the whole partition until the data returns',
			'Saves the warp\'s registers to memory and loads another warp',
			'Asks the CPU for another warp',
			'Issues the next instruction of a different warp that is ready',
		],
		answer: 3,
		why: 'The scheduler hides the wait by issuing from another ready warp. Nothing is saved or restored, because every resident warp keeps its registers in the register file.',
	},
	{
		q: 'Why does switching between resident warps cost nothing?',
		options: [
			'Every resident warp keeps its own registers in the register file the whole time',
			'The L2 cache stores the registers of waiting warps',
			'All warps share the same registers',
			'The switch happens only when a warp finishes',
		],
		answer: 0,
		why: 'Each warp\'s state stays in the register file, so there is nothing to save or restore. This is why the register file is so large.',
	},
	{
		q: 'The register file is 256 KB per SM. How many 32-bit registers is that?',
		options: ['8,192', '65,536', '262,144', '32,768'],
		answer: 1,
		why: 'A 32-bit register is 4 bytes, so 256 * 1024 / 4 = 65,536. 262,144 is the number of bytes, not registers.',
	},
	{
		q: 'Why does a GPU need far more threads than it has cores?',
		options: [
			'Each core can only run one instruction per thread',
			'Threads that finish early are thrown away',
			'With only one warp per partition, little could hide the wait for memory',
			'The L2 cache needs threads to fill it',
		],
		answer: 2,
		why: 'Memory loads take far longer than arithmetic. Many resident warps give the scheduler something ready to issue while others wait.',
	},
	{
		q: 'According to the chapter, why is a single GPU thread slower than a CPU thread?',
		options: [
			'Its arithmetic units run at a much lower precision',
			'It shares its partition with many threads and only gets an instruction issued when it is ready and picked',
			'It has no access to any cache',
			'It must copy its data from the CPU first',
		],
		answer: 1,
		why: 'The arithmetic itself is not slow. Each thread waits its turn in a busy partition, and the total work done per second is what is fast.',
	},
	{
		q: 'Who can see a block\'s shared memory?',
		options: ['One thread', 'Every SM on the chip', 'The CPU, through copies', 'The threads of that block'],
		answer: 3,
		why: 'Shared memory is part of one SM\'s L1 and shared memory block, and the threads of one block share it. Registers are private to one thread, and L2 is visible to every SM.',
	},
	{
		q: 'A T4 has 40 SMs with a 256 KB register file each and a 4 MB L2 cache. How do the registers compare with the L2?',
		options: [
			'The registers total 10 MB, more than the L2',
			'The registers total 256 KB, much less than the L2',
			'The registers total 1 MB, a quarter of the L2',
			'The registers total 4 MB, the same as the L2',
		],
		answer: 0,
		why: '40 * 256 KB = 10,240 KB, which is 10 MB. Added over all SMs, the registers hold more than the L2 cache.',
	},
	{
		q: 'Which statement about a running block is true?',
		options: [
			'It can move between SMs to balance the load',
			'Its threads are spread over several SMs',
			'It runs on one SM and does not move while it runs',
			'It runs on one partition only',
		],
		answer: 2,
		why: 'A block stays on one SM, so its threads can share that SM\'s shared memory and wait for each other.',
	},
	{
		q: 'By default, why can threads in different blocks not share shared memory or wait for each other?',
		options: [
			'Their blocks may be on different SMs',
			'Each block uses a different L2 cache',
			'Blocks always run one after another',
			'Only the CPU can synchronize blocks',
		],
		answer: 0,
		why: 'Shared memory belongs to one SM, and two blocks can land on different SMs. The H100 relaxes this with thread block clusters.',
	},
	{
		q: 'What must an SM have free before it can take one more block?',
		options: [
			'Only a free block slot',
			'An idle partition for every thread in the block',
			'Enough L2 cache for the block\'s data',
			'A free block slot and enough registers, shared memory and warp slots',
		],
		answer: 3,
		why: 'All four resources must fit. Chapter 5 works out how many blocks fit at once.',
	},
	{
		q: 'A kernel launches 100 blocks on an H100 with 132 SMs. At least how many SMs have nothing to do?',
		options: ['0', '32', '100', '66'],
		answer: 1,
		why: 'A block never splits across SMs, so at most 100 SMs can be busy and at least 132 - 100 = 32 are idle.',
	},
	{
		q: 'A T4 SM has 65,536 32-bit registers and at most 32 resident warps. If all those warps are resident, how many registers can each thread use?',
		options: ['32', '128', '64', '2,048'],
		answer: 2,
		why: '32 warps of 32 threads is 1,024 threads, and 65,536 / 1,024 = 64. The answer 32 uses the A100\'s 64 warps per SM instead.',
	},
	{
		q: 'In PyTorch, which field of `torch.cuda.get_device_properties(0)` gives the number of SMs?',
		options: ['`multi_processor_count`', '`L2_cache_size`', '`total_memory`', '`major`'],
		answer: 0,
		why: '`multi_processor_count` is the SM count. `major` and `minor` give the compute capability.',
	},
] satisfies Question[];
