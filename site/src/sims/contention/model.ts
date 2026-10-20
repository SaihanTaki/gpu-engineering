/**
 * A scaled-down histogram launch: THREADS threads in total, split into
 * `blocks` blocks, count N = THREADS x ROUNDS elements into BINS bins with a
 * grid-stride loop. In round r, thread t handles element r * THREADS + t.
 *
 * Two kernels, as in kernels/ch15-histogram/cuda/histogram_kernels.cuh:
 * - global: every element is one atomicAdd on the bin in global memory.
 * - shared: every element is one atomicAdd on the block's private copy in
 *   shared memory; at the end each block adds its nonzero bins to global memory.
 *
 * The model's one rule: atomic operations on the same address are performed
 * one after another, and operations on different addresses proceed in
 * parallel. The CUDA C++ Programming Guide 12.4, section 4.1 (SIMT
 * Architecture), says that when a warp's atomic hits one location from several
 * threads, "each read/modify/write to that location occurs and they are all
 * serialized, but the order in which they occur is undefined". The model applies
 * that to all threads of the step; the different-address case is its
 * simplification. One slot is the time one address takes for one
 * atomic. A step costs as many slots as its longest queue. Slots are a count,
 * not hardware time: a shared memory atomic and a global one cost the same here.
 */
export const BINS = 8;
export const THREADS = 32;
export const ROUNDS = 4;
export const N = THREADS * ROUNDS;
export const BLOCK_OPTIONS = [1, 2, 4, 8] as const;

export const DIST_IDS = ['uniform', 'skewed', 'hot'] as const;
export type Dist = (typeof DIST_IDS)[number];
export const DIST_NAMES: Record<Dist, string> = {
	uniform: 'Uniform: every bin equally likely',
	skewed: 'Skewed: half in bin 0, a quarter in bin 1, ...',
	hot: 'One hot bin: every element in bin 3',
};
export const HOT_BIN = 3;

export const KERNEL_IDS = ['global', 'shared'] as const;
export type KernelId = (typeof KERNEL_IDS)[number];
export const KERNEL_NAMES: Record<KernelId, string> = {
	global: 'Global atomics',
	shared: 'Private copies in shared memory',
};

export interface Config {
	dist: Dist;
	/** Blocks in the grid; THREADS / blocks threads each. */
	blocks: number;
}

/** Steps: 0 start, 1..ROUNDS one grid-stride round each, MERGE_STEP merge, LAST_STEP totals. */
export const MERGE_STEP = ROUNDS + 1;
export const LAST_STEP = ROUNDS + 2;

/** mulberry32, a small seeded generator, so every reader sees the same input. */
function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const cache = new Map<Dist, number[]>();

/** The N bin indices for a distribution. Fixed per distribution. */
export function input(dist: Dist): number[] {
	const hit = cache.get(dist);
	if (hit) return hit;
	const r = rng(15);
	const data = Array.from({ length: N }, () => {
		if (dist === 'hot') return HOT_BIN;
		const u = r();
		if (dist === 'uniform') return Math.floor(u * BINS);
		// Bin b while u < 1 - 2^-(b+1): probability 2^-(b+1), the rest in the last bin.
		let b = 0;
		while (b < BINS - 1 && u >= 1 - 2 ** -(b + 1)) b++;
		return b;
	});
	cache.set(dist, data);
	return data;
}

export const threadsPerBlock = (c: Config) => THREADS / c.blocks;
export const blockOf = (c: Config, thread: number) => Math.floor(thread / threadsPerBlock(c));

/** Bins touched in round r (1-based), one per thread. */
export const roundBins = (c: Config, r: number) => input(c.dist).slice((r - 1) * THREADS, r * THREADS);

/** A queue of atomics waiting on one address, as the threads (or blocks, when merging) that issued them. */
export type Queue = number[];

/** Global kernel, round r: one queue per bin, holding thread ids. */
export function globalQueues(c: Config, r: number): Queue[] {
	const q: Queue[] = Array.from({ length: BINS }, () => []);
	roundBins(c, r).forEach((b, t) => q[b].push(t));
	return q;
}

/** Shared kernel, round r: queues[block][bin], holding thread ids. */
export function sharedQueues(c: Config, r: number): Queue[][] {
	const q: Queue[][] = Array.from({ length: c.blocks }, () => Array.from({ length: BINS }, () => []));
	roundBins(c, r).forEach((b, t) => q[blockOf(c, t)][b].push(t));
	return q;
}

/** Private copy of each block after `rounds` rounds: counts[block][bin]. */
export function privateCounts(c: Config, rounds: number): number[][] {
	const counts = Array.from({ length: c.blocks }, () => new Array(BINS).fill(0));
	for (let r = 1; r <= Math.min(rounds, ROUNDS); r++) roundBins(c, r).forEach((b, t) => counts[blockOf(c, t)][b]++);
	return counts;
}

/** Merge: one queue per global bin, holding the blocks whose private count is nonzero. */
export function mergeQueues(c: Config): Queue[] {
	const counts = privateCounts(c, ROUNDS);
	return Array.from({ length: BINS }, (_, b) => counts.map((row, blk) => (row[b] > 0 ? blk : -1)).filter((blk) => blk >= 0));
}

const longest = (qs: Queue[]) => Math.max(0, ...qs.map((q) => q.length));

/** Slots a step costs for one kernel: its longest queue. */
export function stepSlots(c: Config, k: KernelId, step: number): number {
	if (step >= 1 && step <= ROUNDS) return k === 'global' ? longest(globalQueues(c, step)) : longest(sharedQueues(c, step).flat());
	if (step === MERGE_STEP && k === 'shared') return longest(mergeQueues(c));
	return 0;
}

export interface Totals {
	/** Serialized slots so far: the sum of each step's longest queue. */
	slots: number;
	globalAtomics: number;
	sharedAtomics: number;
	/** Atomics each global bin has served so far. */
	perBin: number[];
}

export function totalsAt(c: Config, k: KernelId, step: number): Totals {
	const t: Totals = { slots: 0, globalAtomics: 0, sharedAtomics: 0, perBin: new Array(BINS).fill(0) };
	for (let s = 1; s <= Math.min(step, MERGE_STEP); s++) t.slots += stepSlots(c, k, s);
	const rounds = Math.min(step, ROUNDS);
	if (k === 'global') {
		for (let r = 1; r <= rounds; r++) roundBins(c, r).forEach((b) => t.perBin[b]++);
		t.globalAtomics = rounds * THREADS;
	} else {
		t.sharedAtomics = rounds * THREADS;
		if (step >= MERGE_STEP) {
			mergeQueues(c).forEach((q, b) => (t.perBin[b] = q.length));
			t.globalAtomics = t.perBin.reduce((a, b) => a + b, 0);
		}
	}
	return t;
}

/** Final histogram; both kernels must produce it. */
export function histogram(dist: Dist): number[] {
	const h = new Array(BINS).fill(0);
	for (const b of input(dist)) h[b]++;
	return h;
}

/** Global bin counts the shared kernel has written after `step`: zero until the merge. */
export function globalHistAt(c: Config, k: KernelId, step: number): number[] {
	if (k === 'shared') return step >= MERGE_STEP ? histogram(c.dist) : new Array(BINS).fill(0);
	const h = new Array(BINS).fill(0);
	for (let r = 1; r <= Math.min(step, ROUNDS); r++) roundBins(c, r).forEach((b) => h[b]++);
	return h;
}

const plural = (n: number, w: string, ws = `${w}s`) => `${n} ${n === 1 ? w : ws}`;
const busiest = (qs: Queue[]) => qs.reduce((best, q, b) => (q.length > qs[best].length ? b : best), 0);

export function describe(c: Config, step: number): string {
	const tpb = threadsPerBlock(c);
	if (step === 0) {
		return `${THREADS} threads in ${plural(c.blocks, 'block')} of ${tpb} count ${N} elements into ${BINS} bins, ${THREADS} elements per round. Guess how many slots each kernel needs, then step.`;
	}
	if (step <= ROUNDS) {
		const g = globalQueues(c, step);
		const gb = busiest(g);
		const s = sharedQueues(c, step);
		const sMax = stepSlots(c, 'shared', step);
		const sBlocks = s.filter((row) => longest(row) === sMax).length;
		return (
			`Round ${step} of ${ROUNDS}. Global: ${THREADS} atomics on ${BINS} counters; bin ${gb} has the longest queue, ${g[gb].length}, so the round takes ${plural(g[gb].length, 'slot')}. ` +
			`Shared: each block's ${tpb} threads use their own copy; the longest queue is ${sMax}${c.blocks > 1 ? `, in ${plural(sBlocks, 'block')}` : ''}, so ${plural(sMax, 'slot')}.`
		);
	}
	if (step === MERGE_STEP) {
		const m = mergeQueues(c);
		const merges = m.reduce((a, q) => a + q.length, 0);
		return `Merge. Global is done. Shared: after __syncthreads each block adds its nonzero bins to global memory, ${plural(merges, 'global atomic')} in all; the longest queue on one bin is ${longest(m)}, so ${plural(longest(m), 'slot')}.`;
	}
	const g = totalsAt(c, 'global', step);
	const s = totalsAt(c, 'shared', step);
	return `Totals, model estimate. Global: ${g.globalAtomics} global atomics, ${plural(g.slots, 'slot')}. Shared: ${s.sharedAtomics} shared and ${plural(s.globalAtomics, 'global atomic')}, ${plural(s.slots, 'slot')}. Both histograms are ${histogram(c.dist).join(', ')}.`;
}

/** Code shown in the sim's panel. Lines are 1-based. */
export const CODE: Record<KernelId, string> = {
	global: `int stride = gridDim.x * blockDim.x;
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    atomicAdd(&hist[data[i]], 1u);`,
	shared: `__shared__ unsigned int local[BINS];
for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
__syncthreads();
int stride = gridDim.x * blockDim.x;
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    atomicAdd(&local[data[i]], 1u);
__syncthreads();
for (int b = threadIdx.x; b < BINS; b += blockDim.x)
    if (local[b] > 0) atomicAdd(&hist[b], local[b]);`,
};

export function lineAt(k: KernelId, step: number): number | null {
	if (step >= 1 && step <= ROUNDS) return k === 'global' ? 3 : 6;
	if (step === MERGE_STEP && k === 'shared') return 9;
	return null;
}
