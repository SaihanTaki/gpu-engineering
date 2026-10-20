/**
 * One thread block summing blockDim.x values, one per thread, with the three
 * shared memory kernels of kernels/ch13-reduction/cuda/reduce_kernels.cuh.
 * The first two follow Harris, "Optimizing Parallel Reduction in CUDA"
 * (https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf),
 * kernels 1 and 3. The third replaces the last six levels with warp shuffles.
 *
 * A warp is 32 threads with consecutive threadIdx.x (CUDA C++ Programming
 * Guide, "SIMT Architecture"). A warp issues an instruction if any of its lanes
 * runs it; lanes that do not are masked off. __shfl_down_sync(mask, v, d) gives
 * lane l the v of lane l + d, or its own v when l + d is past lane 31 (CUDA C++
 * Programming Guide, "Warp Shuffle Functions").
 */
export const WARP = 32;
export const BLOCK_SIZES = [64, 128, 256] as const;

export const MODE_IDS = ['interleaved', 'sequential', 'warp-final'] as const;
export type Mode = (typeof MODE_IDS)[number];
export const MODE_NAMES: Record<Mode, string> = {
	interleaved: 'Interleaved addressing',
	sequential: 'Sequential addressing',
	'warp-final': 'Sequential, shuffle final warp',
};

export interface Config {
	mode: Mode;
	/** Threads per block, a power of two of at least 64. */
	block: number;
}

/** The input element thread t loads: small whole numbers, so every partial sum is readable. */
export const inputValue = (t: number) => (t * 7 + 3) % 10;

const LOOP = (cond: string, head: string) => `sdata[tid] = i < n ? in[i] : 0.0f;
__syncthreads();
${head}
    if (${cond}) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}`;

export const CODE: Record<Mode, string> = {
	interleaved: `${LOOP('tid % (2 * s) == 0', 'for (unsigned s = 1; s < blockDim.x; s *= 2) {')}
if (tid == 0) out[blockIdx.x] = sdata[0];`,
	sequential: `${LOOP('tid < s', 'for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {')}
if (tid == 0) out[blockIdx.x] = sdata[0];`,
	'warp-final': `${LOOP('tid < s', 'for (unsigned s = blockDim.x / 2; s > 32; s /= 2) {')}
if (tid < 32) {
    float v = sdata[tid] + sdata[tid + 32];
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    if (tid == 0) out[blockIdx.x] = v;
}`,
};

export type StepKind = 'load' | 'shared' | 'regs' | 'shuffle' | 'write';

export interface Step {
	kind: StepKind;
	/** Stride s for a shared level, offset for a shuffle, 32 for the register step. */
	stride: number;
	/** Code lines (1-based) this step runs. */
	lines: number[];
	/** Threads that execute this step's add (or load or store). */
	active: boolean[];
	/** Active threads whose result is used later. */
	useful: boolean[];
	/** [destination, source] pairs of the adds that count; register lanes for shuffles. */
	pairs: [number, number][];
	/** After the step: sdata[] and the registers v of warp 0 (null before they exist). */
	shared: number[];
	regs: number[] | null;
	/** After the step: which values are still needed. */
	sharedLive: boolean[];
	regsLive: boolean[] | null;
	/** The block's sum once out[blockIdx.x] is written. */
	out: number | null;
	barrier: boolean;
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

function strides(c: Config): number[] {
	const out: number[] = [];
	if (c.mode === 'interleaved') for (let s = 1; s < c.block; s *= 2) out.push(s);
	else for (let s = c.block / 2; s > (c.mode === 'warp-final' ? 32 : 0); s >>= 1) out.push(s);
	return out;
}

const cache = new Map<string, Step[]>();

/** Every step of the run: the load, each level, and the store of the block's sum. */
export function steps(c: Config): Step[] {
	const key = `${c.mode}:${c.block}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const n = c.block;
	let shared = range(n).map(inputValue);
	let sharedLive = shared.map(() => true);
	let regs: number[] | null = null;
	let regsLive: boolean[] | null = null;
	const all = range(n).map(() => true);
	const out: Step[] = [
		{ kind: 'load', stride: 0, lines: [1, 2], active: all, useful: all, pairs: [], shared, regs, sharedLive, regsLive, out: null, barrier: true },
	];

	for (const s of strides(c)) {
		const active = range(n).map((t) => (c.mode === 'interleaved' ? t % (2 * s) === 0 : t < s));
		const pairs = range(n)
			.filter((t) => active[t])
			.map((t) => [t, t + s] as [number, number]);
		shared = [...shared];
		sharedLive = [...sharedLive];
		for (const [d, src] of pairs) {
			shared[d] += shared[src];
			sharedLive[src] = false;
		}
		out.push({ kind: 'shared', stride: s, lines: [5, 7], active, useful: active, pairs, shared, regs, sharedLive, regsLive, out: null, barrier: true });
	}

	if (c.mode === 'warp-final') {
		const warp0 = range(n).map((t) => t < WARP);
		regs = range(WARP).map((t) => shared[t] + shared[t + WARP]);
		regsLive = regs.map(() => true);
		sharedLive = shared.map(() => false);
		const pairs = range(WARP).flatMap((t) => [[t, t] as [number, number], [t, t + WARP] as [number, number]]);
		out.push({ kind: 'regs', stride: WARP, lines: [10], active: warp0, useful: warp0, pairs, shared, regs, sharedLive, regsLive, out: null, barrier: false });
		for (let o = 16; o > 0; o >>= 1) {
			const prev: number[] = regs;
			regs = prev.map((v, l) => v + (l + o < WARP ? prev[l + o] : v));
			regsLive = range(WARP).map((l) => l < o);
			const useful = range(n).map((t) => t < o);
			const pairs = range(o).map((l) => [l, l + o] as [number, number]);
			out.push({ kind: 'shuffle', stride: o, lines: [12], active: warp0, useful, pairs, shared, regs, sharedLive, regsLive, out: null, barrier: false });
		}
	}

	const first = range(n).map((t) => t === 0);
	const sum = regs ? regs[0] : shared[0];
	out.push({ kind: 'write', stride: 0, lines: [c.mode === 'warp-final' ? 14 : 9], active: first, useful: first, pairs: [], shared, regs, sharedLive, regsLive, out: sum, barrier: false });
	cache.set(key, out);
	return out;
}

export const lastStep = (c: Config) => steps(c).length - 1;
export const stepAt = (c: Config, step: number) => {
	const all = steps(c);
	return all[Math.max(0, Math.min(step, all.length - 1))];
};
export const expectedSum = (c: Config) => range(c.block).reduce((a, t) => a + inputValue(t), 0);

const count = (a: boolean[]) => a.filter(Boolean).length;

/** Active lanes of each warp in a step. */
export const warpLanes = (s: Step) => range(s.active.length / WARP).map((w) => count(s.active.slice(w * WARP, (w + 1) * WARP)));

export type WarpState = 'idle' | 'divergent' | 'full';
export const warpState = (lanes: number): WarpState => (lanes === 0 ? 'idle' : lanes < WARP ? 'divergent' : 'full');

export const isAdd = (s: Step) => s.kind === 'shared' || s.kind === 'regs' || s.kind === 'shuffle';

export interface Totals {
	/** Adds whose result is used: always blockDim.x - 1 at the end. */
	adds: number;
	/** Warp instructions issued for the add steps: one per warp with any lane on. */
	warpIssues: number;
	divergentIssues: number;
	/** __syncthreads() calls inside the reduction loop. */
	barriers: number;
	/** Warp-wide shared memory loads and stores issued by the add steps. */
	sharedRequests: number;
}

/** Totals over steps 1..step. */
export function totalsAt(c: Config, step: number): Totals {
	const t: Totals = { adds: 0, warpIssues: 0, divergentIssues: 0, barriers: 0, sharedRequests: 0 };
	for (const s of steps(c).slice(1, step + 1)) {
		if (!isAdd(s)) continue;
		const lanes = warpLanes(s).filter((l) => l > 0);
		t.warpIssues += lanes.length;
		t.divergentIssues += lanes.filter((l) => l < WARP).length;
		t.adds += s.kind === 'regs' ? count(s.useful) : s.pairs.length;
		if (s.barrier) t.barriers += 1;
		// A shared level reads two words and writes one; the register step reads two.
		t.sharedRequests += lanes.length * (s.kind === 'shared' ? 3 : s.kind === 'regs' ? 2 : 0);
	}
	return t;
}

export const efficiency = (t: Totals) => (t.warpIssues ? t.adds / (t.warpIssues * WARP) : 0);

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** "threads 0, 4, 8, ..., 60" or "threads 0 to 15". */
function threadList(active: boolean[]): string {
	const ids = range(active.length).filter((t) => active[t]);
	if (ids.length === 1) return `thread ${ids[0]}`;
	const contiguous = ids[ids.length - 1] - ids[0] === ids.length - 1;
	if (contiguous) return `threads ${ids[0]} to ${ids[ids.length - 1]}`;
	if (ids.length <= 4) return `threads ${ids.join(', ')}`;
	return `threads ${ids.slice(0, 3).join(', ')}, ..., ${ids[ids.length - 1]}`;
}

function warpSummary(s: Step): string {
	const lanes = warpLanes(s);
	const issuing = lanes.filter((l) => l > 0);
	const div = issuing.filter((l) => l < WARP).length;
	const idle = lanes.length - issuing.length;
	let text = `${plural(issuing.length, 'warp')} of ${lanes.length} ${issuing.length === 1 ? 'issues' : 'issue'} the add`;
	if (div === issuing.length && issuing.length > 0 && issuing.every((l) => l === issuing[0])) {
		text += issuing.length === 1 ? `, with ${issuing[0]} of 32 lanes on (divergent)` : `, each with only ${issuing[0]} of 32 lanes on (all divergent)`;
	} else if (div === 0) text += issuing.length === 1 ? ', with all 32 lanes on' : ', every one with all 32 lanes on';
	else text += `; ${div} of them diverge`;
	if (idle) text += `. ${plural(idle, 'warp')} ${idle === 1 ? 'has no work and skips' : 'have no work and skip'} it`;
	return `${text}.`;
}

export function describe(c: Config, step: number): string {
	const s = stepAt(c, step);
	const all = steps(c);
	const levels = all.filter(isAdd).length;
	const level = all.slice(1, step + 1).filter(isAdd).length;
	const nWarps = c.block / WARP;
	switch (s.kind) {
		case 'load':
			return `Each of the ${c.block} threads (${plural(nWarps, 'warp')}) has loaded one element into sdata[tid], then the block waits at __syncthreads(). The sum takes ${levels} levels. Predict: at the first level, how many warps have work, and how many lanes in each?`;
		case 'shared': {
			const n = count(s.active);
			return `Level ${level} of ${levels}, stride ${s.stride}: ${n === 1 ? 'thread 0 adds' : `${n} threads (${threadList(s.active)}) each add`} sdata[tid + ${s.stride}] into sdata[tid]. ${warpSummary(s)} Then every thread waits at __syncthreads().`;
		}
		case 'regs':
			return `Level ${level} of ${levels}: 64 values are left, so warp 0 alone finishes. Each of its 32 lanes adds sdata[tid] and sdata[tid + 32] into its register v. From here on there is no shared memory traffic and no __syncthreads(); the lanes of one warp exchange registers.`;
		case 'shuffle': {
			const o = s.stride;
			const used = o === 1 ? 'only lane 0' : `lanes 0 to ${o - 1}`;
			return `Level ${level} of ${levels}, shuffle offset ${o}: all 32 lanes of warp 0 run __shfl_down_sync, so the warp does not diverge. Lane l adds the v of lane l + ${o}; the results of ${used} are used, the rest are not. ${nWarps === 2 ? 'Warp 1 left the if and is done.' : `Warps 1 to ${nWarps - 1} left the if and are done.`}`;
		}
		case 'write': {
			const t = totalsAt(c, step);
			return `Done: thread 0 writes the block's sum, ${s.out}, to out[blockIdx.x] (the inputs add up to ${expectedSum(c)}). ${t.adds} useful adds took ${plural(t.warpIssues, 'warp instruction')}, ${t.divergentIssues} of them divergent: ${(100 * efficiency(t)).toFixed(1)}% of the lane slots did useful work. Compare the totals below with the other kernels.`;
		}
	}
}
