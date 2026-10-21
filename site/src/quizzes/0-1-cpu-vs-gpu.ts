import type { Question } from './types';

export default [
	{
		q: 'What does latency measure?',
		options: [
			'How many tasks finish per second',
			'How many workers a machine has',
			'How long one task takes from start to finish',
			'How much data moves per second',
		],
		answer: 2,
		why: 'Latency is the time for one task. Tasks finished per second is throughput, a different number that can be high even when each task is slow.',
	},
	{
		q: 'A shop has twenty checkout lanes with slower cashiers. Compared with one lane and a very fast cashier, what does it have?',
		options: [
			'Higher throughput, even though each customer waits longer at the till',
			'Lower latency, because more customers are served at once',
			'Both lower latency and higher throughput',
			'Lower throughput, because each cashier is slower',
		],
		answer: 0,
		why: 'More lanes means more customers leave per minute, which is throughput. Each slow cashier still takes longer per customer, so latency does not improve.',
	},
	{
		q: 'According to the CUDA Programming Guide, what does a CPU spend more of its transistors on?',
		options: [
			'Many simple arithmetic lanes',
			'A larger number of slow threads',
			'Wider memory buses to DRAM',
			'Data caching and flow control',
		],
		answer: 3,
		why: 'A CPU is built to make one thread fast, so it spends silicon on caches and control logic. Many simple arithmetic lanes is the GPU design.',
	},
	{
		q: 'When does a GPU beat a CPU on a program?',
		options: [
			'Whenever the GPU has more arithmetic units',
			'When there is enough work that can run at the same time to keep its lanes busy',
			'When the program has a large serial part',
			'When each single task must finish as fast as possible',
		],
		answer: 1,
		why: 'Each GPU thread is slower than a CPU thread. The GPU only wins when there is enough parallel work to use its many lanes. A large serial part favors the CPU.',
	},
	{
		q: 'With parallel fraction `p` and `N` workers, what speedup does Amdahl\'s law give?',
		options: [
			'`1 / ((1 - p) + p / N)`',
			'`p * N`',
			'`1 / (p + (1 - p) / N)`',
			'`N / (1 - p)`',
		],
		answer: 0,
		why: 'The serial part `1 - p` takes the same time and the parallel part `p` runs `N` times faster. Swapping `p` and `1 - p` divides the serial part instead of the parallel one.',
	},
	{
		q: 'A program has `p = 0.9`. What speedup does Amdahl\'s law give with 8 workers?',
		options: ['8', 'About 7.2', 'About 4.71', 'About 9.91'],
		answer: 2,
		why: '`1 / (0.1 + 0.9 / 8) = 1 / 0.2125`, about 4.71. The serial 10% keeps it far below 8. About 9.91 is the result for 1000 workers.',
	},
	{
		q: 'A program has `p = 0.9`. What does the speedup approach as the number of workers grows without limit?',
		options: ['9', '10', '90', 'It keeps growing with `N`'],
		answer: 1,
		why: 'As `N` grows, `p / N` goes to 0 and the speedup approaches `1 / (1 - p) = 1 / 0.1 = 10`. The serial part sets a ceiling no number of workers can break.',
	},
	{
		q: 'A program spends 75% of its time in work that splits perfectly. What is the best speedup any number of workers can give?',
		options: ['0.75', '3', '7.5', '4'],
		answer: 3,
		why: 'The limit is `1 / (1 - p) = 1 / 0.25 = 4`. Answers such as 3 or 7.5 come from mixing up `p` and `1 - p` in the formula.',
	},
	{
		q: 'Why can a GPU run the serial part of a program slower than a CPU does?',
		options: [
			'The serial part runs on one of its own threads, which is slow',
			'The GPU splits the serial part across all its lanes',
			'The GPU must copy the serial part to the CPU first',
			'Amdahl\'s law adds a penalty for each extra worker',
		],
		answer: 0,
		why: 'Serial work can only use one worker, and on a GPU that worker is a slow thread. So the GPU does not just fail to speed up the serial part, it runs it slower.',
	},
	{
		q: 'In the race simulation at `p` = 90% with the default settings, which team finishes first?',
		options: [
			'The CPU team',
			'The GPU team',
			'The team that runs the serial part on a CPU worker and the parallel part on the GPU workers',
			'The CPU team and the GPU team tie',
		],
		answer: 2,
		why: 'The third team runs the serial part at CPU speed and the parallel part on many GPU workers, so it is never slower than either of the others. Of the first two, the CPU team wins at 90%.',
	},
	{
		q: 'Why can the speedup the race reaches be a little lower than Amdahl\'s law predicts?',
		options: [
			'The simulation adds a fixed launch cost to every team',
			'The serial part runs twice in the simulation',
			'The GPU workers slow down as more of them are added',
			'The last round of parallel work takes a full round even when only some workers have a unit left',
		],
		answer: 3,
		why: 'Work is handed out in rounds of one unit per worker. A partial last round still costs a whole round, which the formula does not count.',
	},
	{
		q: 'In a CUDA program, what does the host do?',
		options: [
			'Runs the kernels on the GPU',
			'Runs the serial parts, reads input, makes decisions and launches work',
			'Copies data between GPU threads',
			'Compiles kernels while they run',
		],
		answer: 1,
		why: 'The host is the CPU, and it handles the serial parts and launches kernels. The device is the GPU, which runs the kernels.',
	},
	{
		q: 'Which real cost does the race simulation leave out?',
		options: [
			'Moving data between the CPU\'s memory and the GPU\'s memory',
			'The serial part of the job',
			'Running the parallel part in rounds',
			'The slower speed of each GPU worker',
		],
		answer: 0,
		why: 'The simulation models serial work, rounds and slow GPU workers, but not the copy between the two memories. For a job with little arithmetic per element, that copy can take longer than the computation.',
	},
	{
		q: 'What does SAXPY compute?',
		options: ['`y = a * x`', '`y = x * y + a`', '`y = a * x + y`', '`y = a * (x + y)`'],
		answer: 2,
		why: 'SAXPY is single precision `a` times `x` plus `y`, written back into `y`. Every element is independent, so `p` is close to 1.',
	},
	{
		q: 'The CUDA SAXPY uses 256 threads per block and `(n + threads - 1) / threads` blocks. How many blocks does it launch for 1000 elements?',
		options: ['3', '4', '5', '1000'],
		answer: 1,
		why: '`(1000 + 255) / 256 = 1255 / 256`, which is 4 in integer division. Plain `1000 / 256` gives 3 and would leave the last 232 elements uncovered.',
	},
	{
		q: 'Why is the first GPU launch in each SAXPY program not timed?',
		options: [
			'It always produces a wrong result',
			'It runs on the CPU instead of the GPU',
			'It copies the result back, which is timed separately',
			'It includes one-time setup, such as loading the kernel or compiling it',
		],
		answer: 3,
		why: 'The first launch pays costs the later ones do not, such as loading the kernel onto the GPU or, in Numba and Triton, compiling it. Timing it would mix setup into the kernel time.',
	},
	{
		q: 'Which step does `cuda/saxpy.cu` leave out of its timing?',
		options: [
			'The CPU loop',
			'The copy of `x` and `y` to the GPU',
			'The copy of the result back to the CPU',
			'The kernel',
		],
		answer: 2,
		why: 'It times the CPU loop, the copy in and the kernel, but not the copy back. So the real GPU total is higher than the printed numbers.',
	},
	{
		q: 'You want to run `numba/saxpy.py` on a machine with no NVIDIA GPU. What do you set?',
		options: [
			'`NUMBA_ENABLE_CUDASIM=1`, with a small element count',
			'`TRITON_INTERPRET=1`',
			'`CUDA_VISIBLE_DEVICES=0`',
			'Nothing, Numba falls back to the CPU on its own',
		],
		answer: 0,
		why: '`NUMBA_ENABLE_CUDASIM=1` runs Numba kernels on the CPU. `TRITON_INTERPRET=1` is the Triton equivalent. In this mode the GPU times say nothing about real hardware.',
	},
] satisfies Question[];
