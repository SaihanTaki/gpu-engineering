import type { GpuSpec } from '@sim-core/gpu-specs';

export const LEVELS = ['chip', 'sm', 'partition', 'memory'] as const;
export type Level = (typeof LEVELS)[number];
export const lastStep = LEVELS.length - 1;
export const levelAt = (step: number): Level => LEVELS[Math.min(Math.max(0, step), lastStep)];

export interface Component {
	id: string;
	label: string;
	detail: string;
}

export const perPartition = (g: GpuSpec, perSm: number) => perSm / g.partitionsPerSm;

export function totals(g: GpuSpec) {
	return {
		cudaCores: g.sms * g.fp32PerSm,
		tensorCores: g.sms * g.tensorPerSm,
		registerFileKb: g.sms * g.registerFileKbPerSm,
		l1SharedKb: g.sms * g.l1SharedKbPerSm,
		maxWarps: g.sms * g.maxWarpsPerSm,
	};
}

/** Formats a size given in KB using powers of 1024. */
export function formatKb(kb: number): string {
	if (kb >= 1024 * 1024) return `${+(kb / 1024 / 1024).toFixed(1)} GB`;
	if (kb >= 1024) return `${+(kb / 1024).toFixed(1)} MB`;
	return `${kb} KB`;
}

export interface MemoryLevel {
	id: string;
	name: string;
	/** Total capacity on the whole GPU, in KB. */
	totalKb: number;
	scope: string;
}

export function memoryLevels(g: GpuSpec): MemoryLevel[] {
	const t = totals(g);
	return [
		{
			id: 'registers',
			name: 'Registers',
			totalKb: t.registerFileKb,
			scope: `${g.registerFileKbPerSm} KB per SM x ${g.sms} SMs. Private to each thread.`,
		},
		{
			id: 'l1',
			name: 'L1 cache and shared memory',
			totalKb: t.l1SharedKb,
			scope: `${g.l1SharedKbPerSm} KB per SM x ${g.sms} SMs. L1 serves the whole SM. Shared memory is shared by the threads of one block.`,
		},
		{ id: 'l2', name: 'L2 cache', totalKb: g.l2Mb * 1024, scope: 'One cache shared by every SM.' },
		{
			id: 'dram',
			name: `${g.memory} memory`,
			totalKb: g.memoryGb * 1024 * 1024,
			scope: `Off the GPU die, ${g.memoryGbPerS.toLocaleString('en-US')} GB/s to the chip.`,
		},
	];
}

export function components(g: GpuSpec, level: Level): Component[] {
	const t = totals(g);
	const pp = (n: number) => perPartition(g, n);
	switch (level) {
		case 'chip':
			return [
				{
					id: 'sms',
					label: `${g.sms} SMs`,
					detail: `Streaming multiprocessors do all the computing. Each one runs its own blocks of threads independently. This ${g.name} has ${g.sms}, for ${t.cudaCores.toLocaleString('en-US')} FP32 cores in total.`,
				},
				{
					id: 'l2',
					label: `L2 cache, ${g.l2Mb} MB`,
					detail: `Every SM reads and writes ${g.memory} memory through the L2 cache, so the L2 is shared by the whole chip.`,
				},
				{
					id: 'dram',
					label: `${g.memoryGb} GB ${g.memory}`,
					detail: `The GPU's main memory, sometimes called global memory. It sits next to the GPU die, not on it, and delivers ${g.memoryGbPerS.toLocaleString('en-US')} GB/s.`,
				},
			];
		case 'sm':
			return [
				{
					id: 'partitions',
					label: `${g.partitionsPerSm} partitions`,
					detail: `An SM is split into ${g.partitionsPerSm} partitions. Each has its own warp scheduler, a slice of the register file and its own execution units.`,
				},
				{
					id: 'l1',
					label: `L1 / shared, ${g.l1SharedKbPerSm} KB`,
					detail: `One block of on-chip memory serves as both the L1 data cache and shared memory. A program can set aside up to ${g.maxSharedKbPerSm} KB of it as shared memory.`,
				},
				{
					id: 'warps',
					label: `up to ${g.maxWarpsPerSm} warps`,
					detail: `Threads run in warps of 32. Up to ${g.maxWarpsPerSm} warps (${g.maxWarpsPerSm * 32} threads) can be resident on one SM at a time.`,
				},
			];
		case 'partition':
			return [
				{
					id: 'scheduler',
					label: 'Warp scheduler',
					detail: `At each issue slot the scheduler picks one warp that is ready and issues its next instruction. Warps waiting on memory are skipped, which is how the SM hides memory latency.`,
				},
				{
					id: 'registers',
					label: `Registers, ${pp(g.registerFileKbPerSm)} KB`,
					detail: `${pp(g.registerFileKbPerSm)} KB is ${(pp(g.registerFileKbPerSm) * 256).toLocaleString('en-US')} 32-bit registers, shared by all the warps assigned to this partition.`,
				},
				{
					id: 'fp32',
					label: `${pp(g.fp32PerSm)} FP32 cores`,
					detail: `What NVIDIA calls CUDA cores. Each does one 32-bit floating point operation per clock, or a fused multiply-add.`,
				},
				{
					id: 'int32',
					label: `${pp(g.int32PerSm)} INT32 units`,
					detail: 'Integer math: index arithmetic, addresses, loop counters.',
				},
				{
					id: 'fp64',
					label: Number.isInteger(pp(g.fp64PerSm))
						? `${pp(g.fp64PerSm)} FP64 units`
						: `FP64: ${g.fp64PerSm} per SM`,
					detail: Number.isInteger(pp(g.fp64PerSm))
						? 'Double-precision floating point, used mostly in scientific computing.'
						: `Double-precision floating point. ${g.architecture} has only ${g.fp64PerSm} FP64 units per whole SM, so FP64 code runs far slower than FP32 on this GPU.`,
				},
				{
					id: 'tensor',
					label: `${pp(g.tensorPerSm)} tensor core${pp(g.tensorPerSm) === 1 ? '' : 's'}`,
					detail: `${g.tensorGeneration[0].toUpperCase()}${g.tensorGeneration.slice(1)}-generation tensor cores multiply small matrices in one operation. Chapter 23 programs them.`,
				},
				{
					id: 'sfu',
					label: `${pp(g.sfuPerSm)} SFUs`,
					detail: 'Special function units compute fast approximations of reciprocal, reciprocal square root, base-2 exponent and logarithm, sine and cosine.',
				},
			];
		case 'memory':
			return memoryLevels(g).map((m) => ({
				id: m.id,
				label: `${m.name}: ${formatKb(m.totalKb)}`,
				detail: m.scope,
			}));
	}
}

export function describe(g: GpuSpec, step: number): string {
	const t = totals(g);
	switch (levelAt(step)) {
		case 'chip':
			return `The whole ${g.name}: ${g.sms} SMs share one ${g.l2Mb} MB L2 cache and ${g.memoryGb} GB of ${g.memory}. Press Next to zoom into one SM.`;
		case 'sm':
			return `One SM: ${g.partitionsPerSm} partitions, ${g.fp32PerSm} FP32 cores and ${g.tensorPerSm} tensor cores, with ${g.l1SharedKbPerSm} KB of L1 cache and shared memory. Press Next to zoom into one partition.`;
		case 'partition':
			return `One partition: a warp scheduler, ${perPartition(g, g.registerFileKbPerSm)} KB of registers and ${perPartition(g, g.fp32PerSm)} FP32 cores. Press Next to compare memory sizes.`;
		case 'memory': {
			const l2Kb = g.l2Mb * 1024;
			const compare =
				t.registerFileKb > l2Kb
					? `the registers (${formatKb(t.registerFileKb)}) hold more than the L2 cache (${formatKb(l2Kb)})`
					: `the registers add up to ${formatKb(t.registerFileKb)}, against ${formatKb(l2Kb)} of L2 cache`;
			return `Memory on the whole ${g.name}: added over all SMs, ${compare}. The fastest storage is spread across the SMs, not in one place.`;
		}
	}
}
