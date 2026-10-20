import type { GpuSpec } from '@sim-core/gpu-specs';

/**
 * Theoretical occupancy of one SM, following cudaOccMaxActiveBlocksPerMultiprocessor
 * in cuda_occupancy.h (CUDA Toolkit) for compute capability 7.5, 8.0 and 9.0.
 * Assumes the default shared memory carveout (all of maxSharedKbPerSm usable) and
 * that the kernel has opted in to the largest dynamic shared memory size.
 */
export const WARP = 32;

export interface Kernel {
	regsPerThread: number;
	/** Static plus dynamic shared memory the kernel asks for, in bytes. */
	smemPerBlock: number;
	blockSize: number;
}

export const RESOURCES = ['warps', 'registers', 'shared', 'blocks'] as const;
export type ResourceId = (typeof RESOURCES)[number];

export const RESOURCE_NAMES: Record<ResourceId, string> = {
	warps: 'warp slots',
	registers: 'registers',
	shared: 'shared memory',
	blocks: 'block slots',
};

export interface Occupancy {
	warpsPerBlock: number;
	/** Registers one warp is given, after rounding up to regAllocUnit. */
	regsPerWarp: number;
	regsPerBlock: number;
	regsPerSm: number;
	regsPerPartition: number;
	/** Warps whose registers fit in one partition. */
	warpsPerPartition: number;
	/** Shared memory one block takes, reserved part included, after rounding. */
	smemPerBlock: number;
	smemPerSm: number;
	/** Blocks per SM each resource allows; Infinity when the resource is not used. */
	limits: Record<ResourceId, number>;
	blocks: number;
	activeWarps: number;
	/** Active warps / maximum warps per SM, 0..1. */
	occupancy: number;
	limiting: ResourceId[];
	/** Why the launch fails, or null if at least one block fits. */
	error: string | null;
}

const roundUp = (x: number, unit: number) => Math.ceil(x / unit) * unit;
export const fmt = (n: number) => n.toLocaleString('en-US');
export const kb = (bytes: number) => `${+(bytes / 1024).toFixed(2)} KB`;
export const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
export const pct = (x: number) => `${+(100 * x).toFixed(2)}%`;

export function occupancy(g: GpuSpec, k: Kernel): Occupancy {
	const warpsPerBlock = Math.ceil(k.blockSize / WARP);
	const regsPerSm = (g.registerFileKbPerSm * 1024) / 4;
	const regsPerPartition = regsPerSm / g.partitionsPerSm;
	const regsPerWarp = roundUp(k.regsPerThread * WARP, g.regAllocUnit);
	const regsPerBlock = regsPerWarp * warpsPerBlock;
	// The hardware check assumes a block's warps are spread over every partition.
	const regsAssumed = regsPerWarp * roundUp(warpsPerBlock, g.partitionsPerSm);
	const smemPerSm = g.maxSharedKbPerSm * 1024;
	const smemPerBlock = roundUp(k.smemPerBlock + g.reservedSmemPerBlock, g.smemAllocUnit);
	const smemLimitPerBlock = g.maxSharedKbPerBlock * 1024 + g.reservedSmemPerBlock;
	const warpsPerPartition = regsPerWarp > 0 ? Math.floor(regsPerPartition / regsPerWarp) : Infinity;

	let error: string | null = null;
	if (k.blockSize > g.maxThreadsPerBlock) error = `A block can have at most ${g.maxThreadsPerBlock} threads.`;
	else if (k.regsPerThread > g.maxRegsPerThread) error = `A thread can use at most ${g.maxRegsPerThread} registers.`;

	const warps = k.blockSize > g.maxThreadsPerBlock ? 0 : Math.floor(g.maxWarpsPerSm / warpsPerBlock);
	let registers: number;
	if (regsAssumed > g.maxRegsPerBlock || k.regsPerThread > g.maxRegsPerThread) {
		registers = 0;
		error ??= `The block needs ${fmt(regsAssumed)} registers (${fmt(regsPerWarp)} per warp x ${roundUp(warpsPerBlock, g.partitionsPerSm)} warps, rounded up to a multiple of ${g.partitionsPerSm}), more than the ${fmt(g.maxRegsPerBlock)} a block may use. The launch fails with "too many resources requested for launch".`;
	} else {
		registers = regsPerWarp > 0 ? Math.floor((warpsPerPartition * g.partitionsPerSm) / warpsPerBlock) : Infinity;
	}
	let shared: number;
	if (smemPerBlock > smemLimitPerBlock) {
		shared = 0;
		error ??= `The block needs ${kb(smemPerBlock)} of shared memory, more than the ${g.maxSharedKbPerBlock} KB a block can have on the ${g.name}.`;
	} else {
		shared = smemPerBlock > 0 ? Math.floor(smemPerSm / smemPerBlock) : Infinity;
	}
	const limits = { warps, registers, shared, blocks: g.maxBlocksPerSm };
	const blocks = Math.min(...RESOURCES.map((r) => limits[r]));
	if (blocks === 0 && !error) error = 'Not even one block fits on an SM.';
	const activeWarps = blocks * warpsPerBlock;
	return {
		warpsPerBlock,
		regsPerWarp,
		regsPerBlock,
		regsPerSm,
		regsPerPartition,
		warpsPerPartition,
		smemPerBlock,
		smemPerSm,
		limits,
		blocks,
		activeWarps,
		occupancy: activeWarps / g.maxWarpsPerSm,
		limiting: RESOURCES.filter((r) => limits[r] === blocks),
		error,
	};
}

/** Step 0 is the empty SM, step b places block b - 1, and the last step tries one block too many. */
export const lastStep = (o: Occupancy) => (o.blocks > 0 ? o.blocks + 1 : 0);

/** Blocks resident at a step. */
export const placedAt = (o: Occupancy, step: number) => Math.min(Math.max(0, step), o.blocks);

/** True at the step that tries to add a block that does not fit. */
export const overflowAt = (o: Occupancy, step: number) => o.blocks > 0 && step > o.blocks;

/**
 * Partition of each warp. Warps are dealt to partitions in turn, which always
 * fits because blocks * warpsPerBlock <= partitions * warpsPerPartition.
 */
export const partitionOf = (g: GpuSpec, warpIndex: number) => warpIndex % g.partitionsPerSm;

/** Resources used by `placed` blocks, for the bars and the text. */
export function usage(g: GpuSpec, o: Occupancy, placed: number) {
	const warps = placed * o.warpsPerBlock;
	const perPartition = Array.from({ length: g.partitionsPerSm }, (_, p) =>
		Math.max(0, Math.ceil((warps - p) / g.partitionsPerSm)),
	);
	return {
		warps,
		regs: warps * o.regsPerWarp,
		/** Warps placed in each partition. */
		perPartition,
		smem: placed * o.smemPerBlock,
		blocks: placed,
	};
}

/** Why the next block does not fit on this resource, or null if it would. */
export function shortfall(g: GpuSpec, o: Occupancy, r: ResourceId): string | null {
	if (o.limits[r] > o.blocks) return null;
	const u = usage(g, o, o.blocks);
	switch (r) {
		case 'warps':
			return `${g.maxWarpsPerSm - u.warps} of ${g.maxWarpsPerSm} warp slots are free and a block needs ${o.warpsPerBlock}`;
		case 'registers': {
			const freeWarps = u.perPartition.map((w) => o.warpsPerPartition - w);
			const free = freeWarps.reduce((a, b) => a + b, 0);
			return `the four partitions have room for ${free} more warp${free === 1 ? '' : 's'} of ${fmt(o.regsPerWarp)} registers and a block needs ${o.warpsPerBlock}`;
		}
		case 'shared':
			return `${kb(o.smemPerSm - u.smem)} of shared memory is free and a block needs ${kb(o.smemPerBlock)}`;
		case 'blocks':
			return `all ${g.maxBlocksPerSm} block slots are taken`;
	}
}

export function describe(g: GpuSpec, k: Kernel, step: number): string {
	const o = occupancy(g, k);
	if (o.error) return `${o.error} Theoretical occupancy is 0%.`;
	const placed = placedAt(o, step);
	const u = usage(g, o, placed);
	if (step === 0) {
		const smem = g.reservedSmemPerBlock
			? `${kb(o.smemPerBlock)} of shared memory (${kb(k.smemPerBlock)} asked for plus ${kb(g.reservedSmemPerBlock)} reserved, rounded up to ${g.smemAllocUnit} bytes)`
			: `${kb(o.smemPerBlock)} of shared memory`;
		return `An empty SM on the ${g.name}. Each block of ${k.blockSize} threads needs ${o.warpsPerBlock} warp slot${o.warpsPerBlock === 1 ? '' : 's'}, ${fmt(o.regsPerBlock)} registers (${fmt(o.regsPerWarp)} per warp), ${smem} and one block slot. Predict how many blocks fit, then press Next.`;
	}
	if (overflowAt(o, step)) {
		const reasons = o.limiting.map((r) => `${cap(RESOURCE_NAMES[r])}: ${shortfall(g, o, r)}`);
		return `Block ${o.blocks} does not fit. ${reasons.join('. ')}. Theoretical occupancy: ${plural(o.blocks, 'block')} x ${plural(o.warpsPerBlock, 'warp')} = ${o.activeWarps} of ${g.maxWarpsPerSm} warps, ${pct(o.occupancy)}. Limited by ${o.limiting.map((r) => RESOURCE_NAMES[r]).join(' and ')}.`;
	}
	return `Block ${placed - 1} placed: ${placed} block${placed === 1 ? '' : 's'}, ${u.warps} of ${g.maxWarpsPerSm} warp slots, ${fmt(u.regs)} of ${fmt(o.regsPerSm)} registers, ${kb(u.smem)} of ${g.maxSharedKbPerSm} KB shared memory. Occupancy so far ${pct(u.warps / g.maxWarpsPerSm)}.`;
}
