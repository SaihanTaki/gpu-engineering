import type { Question } from './types';

export default [
	{
		q: 'Run eagerly, how many GPU kernels does PyTorch launch for `out = torch.relu(a * x + y)`?',
		options: ['1', '2', '3', '4'],
		answer: 2,
		why: 'One kernel multiplies, one adds and one clamps. Each reads its inputs from DRAM and writes its result back.',
	},
	{
		q: 'Which of these is not a map?',
		options: ['ReLU', 'Casting every element to another type', 'Adding a bias to every element', 'A running total of the array'],
		answer: 3,
		why: 'In a map, output element `i` depends only on input elements at index `i`. A running total depends on every earlier element.',
	},
	{
		q: 'The fused kernel reads `x[i]` and `y[i]`, writes `out[i]`, and does a multiply, an add and a compare. How many operations per byte is that?',
		options: ['0.25', '0.5', '3', '4'],
		answer: 0,
		why: '3 operations over 12 bytes (three 4-byte floats) is 0.25. Counting only the 8 bytes read would give the wrong 0.375.',
	},
	{
		q: 'The A100 does 19.5 TFLOPS of FP32 and moves 2,039 GB/s. About how many FP32 operations per byte does a kernel need to keep the arithmetic units busy?',
		options: ['0.1', '1', '96', '9.6'],
		answer: 3,
		why: '19.5 x 10^12 / 2.039 x 10^12 is about 9.6. At 0.25 operations per byte, the fused kernel is far below that and memory-bound.',
	},
	{
		q: 'A kernel is memory-bound. Which change makes it faster?',
		options: [
			'Replacing a multiply and an add with one FMA',
			'Doing the arithmetic in fewer instructions',
			'Moving fewer bytes to and from DRAM',
			'Launching more threads per block',
		],
		answer: 2,
		why: 'A memory-bound kernel spends its time waiting on bytes, so less arithmetic does not help. Fusion helps because it removes bytes.',
	},
	{
		q: 'The unfused version runs `scale`, `add` and `relu`. How many arrays worth of bytes do they move in total?',
		options: ['3', '5', '6', '7'],
		answer: 3,
		why: '`scale` moves 2 arrays, `add` 3 and `relu` 2, so 7. The fused kernel moves 3.',
	},
	{
		q: 'Of the 7 arrays the unfused version moves, how many are traffic for intermediate values?',
		options: ['2', '4', '3', '5'],
		answer: 1,
		why: '`t` and `u` are each written once and read once: 4 array moves for values that exist only to pass to the next kernel.',
	},
	{
		q: 'For `2^24` floats, how many bytes does the fused kernel move?',
		options: ['64 MiB', '192 MiB', '256 MiB', '448 MiB'],
		answer: 1,
		why: 'Each array is `2^24` x 4 bytes = 64 MiB, and the fused kernel moves 3 of them. 448 MiB is the unfused total of 7 arrays.',
	},
	{
		q: 'A chain of 4 maps runs `x -> t1 -> t2 -> t3 -> out`, one kernel per step, each reading one array and writing one. What is the traffic ratio of unfused to fused?',
		options: ['2', '3', '4', '8'],
		answer: 2,
		why: 'Unfused moves 4 x 2 = 8 arrays. Fused reads `x` and writes `out`, 2 arrays. Each of the 3 intermediates costs one write and one read.',
	},
	{
		q: 'Where does the simulation get its default launch cost of 1 µs?',
		options: [
			'Rounded from the 0.9 µs gap in a V100 measurement in the CUDA Graphs post',
			'From an NVIDIA specification for all GPUs',
			'From the 2.9 µs kernel time in that measurement, divided by three',
			'From the A100 datasheet',
		],
		answer: 0,
		why: 'A 2.9 µs kernel took 3.8 µs per kernel back to back, a 0.9 µs gap. NVIDIA publishes no single launch cost figure.',
	},
	{
		q: 'In the model, at what element count does the fused kernel memory time on a Tesla T4 (300 GB/s) equal one 1 µs launch?',
		options: ['25,000', '170,000', '2,500', '300,000'],
		answer: 0,
		why: '12 n / (300 x 10^9) = 10^-6 gives n = 25,000. About 170,000 is the A100 answer, where memory is faster.',
	},
	{
		q: 'At `2^24` elements on the A100, the model gives a time ratio of 2.34, just above the 7 / 3 = 2.33 traffic ratio. What adds the extra?',
		options: [
			'Arithmetic time in the unfused kernels',
			'Three launches against one',
			'L2 misses on the intermediates',
			'The FMA rounding difference',
		],
		answer: 1,
		why: 'The model is bytes at peak bandwidth plus a launch cost per kernel. It counts no arithmetic and no cache.',
	},
	{
		q: 'At the smallest size, `2^10` elements, what does the model ratio approach?',
		options: ['1', '2.33', '7', '3'],
		answer: 3,
		why: 'Each array is 4 KiB and moves in nanoseconds, so launches are nearly all the time: 3 launches against 1.',
	},
	{
		q: 'With the launch cost slider at 0, what ratio does the model give?',
		options: [
			'Exactly 7 / 3 at every size',
			'3 at every size',
			'7 / 3 at large sizes and 3 at small sizes',
			'1, because both versions take no time',
		],
		answer: 0,
		why: 'Without launch cost the model is pure traffic, and the traffic ratio is 7 / 3 at any size.',
	},
	{
		q: 'How does the model err for a kernel that moves only a few kilobytes?',
		options: [
			'It overestimates the time, because it charges the launch twice',
			'It ignores the bytes entirely',
			'It underestimates the time, because real kernels that small never reach peak bandwidth',
			'It is exact, because peak bandwidth holds at every size',
		],
		answer: 2,
		why: 'Too few warps are in flight to hide DRAM latency, so real bandwidth is below peak. The model charges peak bandwidth at every size.',
	},
	{
		q: 'The fused kernel PTX contains how many `ld.global.f32` instructions?',
		options: ['1', '2', '3', '4'],
		answer: 1,
		why: 'It loads `x[i]` and `y[i]`, does one `fma.rn.f32` and one `max.f32`, and stores once. Nothing in between touches memory.',
	},
	{
		q: 'Why can the fused and unfused results differ in the last bit?',
		options: [
			'The fused kernel uses half precision for the intermediate',
			'ReLU is computed with a different rounding mode',
			'The unfused version reads stale data from L2',
			'The fused multiply-add rounds once, while `scale` stores a rounded `a * x` before `add` sees it',
		],
		answer: 3,
		why: '`nvcc` and NVRTC contract the multiply and add into one FMA by default. That is why the tests compare with a small tolerance.',
	},
	{
		q: 'Which compiler option controls the contraction of a multiply and an add into an FMA, and what is its default?',
		options: ['`--fmad=false`, off by default', '`--fmad=true`, on by default', '`--use_fast_math`, off by default', '`-O3`, on by default'],
		answer: 1,
		why: 'The default is `--fmad=true`, so a multiply followed by an add becomes one fused multiply-add unless you turn it off.',
	},
	{
		q: 'You want to see the three kernels of one unfused pass and the gaps between them. Which tool do you use?',
		options: [
			'`ncu --metrics dram__bytes_read.sum`',
			'`compute-sanitizer`',
			'`nsys profile --stats=true`',
			'`cuobjdump`',
		],
		answer: 2,
		why: 'The Nsight Systems timeline shows kernels and gaps. Nsight Compute with the DRAM metrics counts bytes per kernel.',
	},
	{
		q: 'What generates fused Triton kernels from PyTorch code without writing them by hand?',
		options: ['`torch.compile` with its default TorchInductor backend', 'CUDA Graphs', 'NVRTC', 'Running PyTorch eagerly'],
		answer: 0,
		why: 'TorchInductor generates Triton kernels and fuses elementwise operations. CUDA Graphs cut launch cost but do not fuse.',
	},
] satisfies Question[];
