import { gpuById, type GpuSpec } from '@sim-core/gpu-specs';

/**
 * One warp of a 32 x 8 block moving one row of a 32 x 32 tile through the three
 * transpose kernels of Harris, "An Efficient Matrix Transpose in CUDA C/C++"
 * (https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).
 * The matrix is n x n float32, row-major, starting at a 256-byte aligned address.
 *
 * Global memory: a warp's request is served in 32-byte sectors; it costs as many
 * sectors as distinct 32-byte segments its active lanes touch (CUDA C++ Best
 * Practices Guide, "Coalesced Access to Global Memory").
 * Shared memory: 32 banks, successive 32-bit words in successive banks, each bank
 * 32 bits per clock. Lanes reading the same word are served at once; distinct words
 * in one bank are serialized (CUDA Programming Guide, shared memory of compute
 * capability 5.x and later). A request costs as many passes as the most distinct
 * words any bank receives.
 */
export const WARP = 32;
export const TILE = 32;
export const WORD = 4;
export const SECTOR = 32;
export const BANKS = 32;
export const WORDS_PER_SECTOR = SECTOR / WORD;

export const KERNEL_IDS = ['naive', 'tiled', 'padded'] as const;
export type KernelId = (typeof KERNEL_IDS)[number];
export const KERNEL_NAMES: Record<KernelId, string> = {
	naive: 'Naive',
	tiled: 'Tiled',
	padded: 'Tiled + padding',
};
/** Row pitch of the shared tile in words. */
export const PITCH: Record<KernelId, number> = { naive: 0, tiled: TILE, padded: TILE + 1 };

export const SIZES = [1024, 1000, 1001] as const;

export interface Config {
	gpu: GpuSpec['id'];
	/** Matrix is n x n. */
	n: number;
	tile: 'first' | 'last';
	/** Tile row this warp moves, threadIdx.y + j, 0..31. */
	row: number;
}

export type AccessKind = 'global-load' | 'global-store' | 'shared-store' | 'shared-load';

export interface Access {
	kind: AccessKind;
	/** Element index (global) or word index in the tile (shared) per lane; null when the bounds check fails. */
	index: (number | null)[];
}

export interface SectorUse {
	sector: number;
	/** Which of the sector's 8 words the warp uses. */
	words: boolean[];
}

export interface GlobalCost {
	sectors: SectorUse[];
	usedBytes: number;
}

export interface SharedCost {
	/** Distinct words each bank is asked for. */
	perBank: number[];
	/** Passes the request needs: the largest perBank, 0 if no lane is active. */
	degree: number;
}

const active = (a: Access) => a.index.filter((i): i is number => i !== null);

export function globalCost(a: Access): GlobalCost {
	const bySector = new Map<number, boolean[]>();
	for (const i of active(a)) {
		const s = Math.floor((i * WORD) / SECTOR);
		const words = bySector.get(s) ?? new Array(WORDS_PER_SECTOR).fill(false);
		words[i % WORDS_PER_SECTOR] = true;
		bySector.set(s, words);
	}
	const sectors = [...bySector].sort((x, y) => x[0] - y[0]).map(([sector, words]) => ({ sector, words }));
	return { sectors, usedBytes: active(a).length * WORD };
}

export function sharedCost(a: Access): SharedCost {
	const words = new Set(active(a));
	const perBank = new Array(BANKS).fill(0);
	for (const w of words) perBank[w % BANKS]++;
	return { perBank, degree: Math.max(0, ...perBank) };
}

export const tilesPerSide = (n: number) => Math.ceil(n / TILE);

/** Block indices (bx, by) of the chosen tile; the first and last tiles lie on the diagonal. */
export function tileOf(c: Config) {
	const t = c.tile === 'first' ? 0 : tilesPerSide(c.n) - 1;
	return { bx: t, by: t };
}

const lanes = (f: (lane: number) => number | null) => Array.from({ length: WARP }, (_, lane) => f(lane));

/**
 * The accesses one warp makes, in program order. Phase A reads row `row` of the
 * input tile; phase B writes row `row` of the mirrored output tile.
 */
export function accesses(c: Config, k: KernelId): Access[] {
	const { bx, by } = tileOf(c);
	const { n, row: r } = c;
	// Phase A: x = bx * 32 + lane, y + j = by * 32 + r.
	const okA = (lane: number) => bx * TILE + lane < n && by * TILE + r < n;
	const load: Access = { kind: 'global-load', index: lanes((l) => (okA(l) ? (by * TILE + r) * n + bx * TILE + l : null)) };
	if (k === 'naive') {
		return [load, { kind: 'global-store', index: lanes((l) => (okA(l) ? (bx * TILE + l) * n + by * TILE + r : null)) }];
	}
	const p = PITCH[k];
	// Phase B: x = by * 32 + lane, y + j = bx * 32 + r.
	const okB = (lane: number) => by * TILE + lane < n && bx * TILE + r < n;
	return [
		load,
		{ kind: 'shared-store', index: lanes((l) => (okA(l) ? r * p + l : null)) },
		{ kind: 'shared-load', index: lanes((l) => (okB(l) ? l * p + r : null)) },
		{ kind: 'global-store', index: lanes((l) => (okB(l) ? (bx * TILE + r) * n + by * TILE + l : null)) },
	];
}

/**
 * Steps: 0 start, 1 global load, 2 first store, 3 __syncthreads, 4 shared load,
 * 5 global store, 6 totals and estimate. The naive kernel finishes at step 2.
 */
export const LAST_STEP = 6;

/** Index into accesses(c, k) shown at a step, or null when the kernel does nothing then. */
export function accessIndexAt(k: KernelId, step: number): number | null {
	if (step < 1 || step > 5) return null;
	if (k === 'naive') return step <= 2 ? step - 1 : null;
	if (step === 3) return null;
	return step <= 2 ? step - 1 : step - 2;
}

export const isSync = (k: KernelId, step: number) => k !== 'naive' && step === 3;

export interface Totals {
	loadSectors: number;
	storeSectors: number;
	sharedPasses: number;
	/** Bytes the warp actually needs: every active lane's 4 bytes, read once and written once. */
	usefulBytes: number;
	/** Bytes of the sectors moved. */
	sectorBytes: number;
}

/** Counts over the accesses done by the end of `step`. */
export function totalsAt(c: Config, k: KernelId, step: number): Totals {
	const acc = accesses(c, k);
	const t: Totals = { loadSectors: 0, storeSectors: 0, sharedPasses: 0, usefulBytes: 0, sectorBytes: 0 };
	for (let s = 1; s <= Math.min(step, 5); s++) {
		const i = accessIndexAt(k, s);
		if (i === null) continue;
		const a = acc[i];
		if (a.kind === 'global-load' || a.kind === 'global-store') {
			const g = globalCost(a);
			if (a.kind === 'global-load') t.loadSectors += g.sectors.length;
			else t.storeSectors += g.sectors.length;
			t.sectorBytes += g.sectors.length * SECTOR;
			t.usefulBytes += g.usedBytes;
		} else {
			t.sharedPasses += sharedCost(a).degree;
		}
	}
	return t;
}

/**
 * SM clock in GHz, derived from each preset's peak FP32 rate:
 * TFLOPS / (SMs x FP32 lanes per SM x 2 flops per FMA). Peak FP32 from the
 * T4 datasheet (8.1 TFLOPS), A100 datasheet (19.5 TFLOPS) and H100 SXM product
 * page (67 TFLOPS), the sources listed in gpu-specs.ts.
 */
export const PEAK_FP32_TFLOPS: Record<GpuSpec['id'], number> = { t4: 8.1, a100: 19.5, h100: 67 };
export const clockGhz = (g: GpuSpec) => (PEAK_FP32_TFLOPS[g.id] * 1e3) / (g.sms * g.fp32PerSm * 2);

export interface Estimate {
	/** Effective bandwidth, GB/s: useful bytes over modeled time. Null when the warp moves nothing. */
	gbPerS: number | null;
	limit: 'global' | 'shared' | null;
	globalNs: number;
	sharedNs: number;
}

/**
 * Model estimate, not a measurement. Every warp in the GPU is assumed to do what
 * this one does. Sector traffic is charged at the preset's memory bandwidth; each
 * shared memory pass takes one clock of one SM's shared memory; the two overlap,
 * so the slower one sets the time.
 */
export function estimate(g: GpuSpec, t: Totals): Estimate {
	const globalNs = t.sectorBytes / g.memoryGbPerS;
	const sharedNs = t.sharedPasses / (g.sms * clockGhz(g));
	if (t.usefulBytes === 0) return { gbPerS: null, limit: null, globalNs, sharedNs };
	const limit = sharedNs > globalNs ? 'shared' : 'global';
	return { gbPerS: t.usefulBytes / Math.max(globalNs, sharedNs), limit, globalNs, sharedNs };
}

export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
const plural = (n: number, w: string, ws = `${w}s`) => `${n} ${n === 1 ? w : ws}`;
const lanesOn = (a: Access) => active(a).length;

function describeAccess(a: Access, k: KernelId): string {
	const on = lanesOn(a);
	if (on === 0) return 'every lane fails the bounds check, so nothing is accessed';
	const masked = on < WARP ? ` (${on} lanes pass the bounds check)` : '';
	if (a.kind === 'global-load' || a.kind === 'global-store') {
		const g = globalCost(a);
		const what = a.kind === 'global-load' ? 'reads a row of in' : k === 'naive' ? 'writes a column of out' : 'writes a row of out';
		return `${what}${masked}: ${plural(g.sectors.length, 'sector')}, ${fmt(g.sectors.length * SECTOR)} bytes moved for ${fmt(g.usedBytes)} used`;
	}
	const s = sharedCost(a);
	const what = a.kind === 'shared-store' ? 'writes a row of the tile' : 'reads a column of the tile';
	return `${what}${masked}: ${s.degree <= 1 ? 'no bank conflict, 1 pass' : `${s.degree}-way bank conflict, ${s.degree} passes`}`;
}

export function describeKernel(c: Config, k: KernelId, step: number): string {
	const name = KERNEL_NAMES[k];
	if (isSync(k, step)) return `${name} waits at __syncthreads until all 8 warps of the block have written their rows.`;
	const i = accessIndexAt(k, step);
	if (i === null) return `${name} is done.`;
	return `${name} ${describeAccess(accesses(c, k)[i], k)}.`;
}

export function describe(c: Config, step: number): string {
	const g = gpuById(c.gpu);
	if (step === 0) {
		return `One warp, 32 lanes, moves row ${c.row} of the ${c.tile} tile of a ${c.n} x ${c.n} matrix through each kernel. Guess which kernel moves the most sectors, then step.`;
	}
	if (step < LAST_STEP) return KERNEL_IDS.map((k) => describeKernel(c, k, step)).join(' ');
	return KERNEL_IDS.map((k) => {
		const t = totalsAt(c, k, step);
		const e = estimate(g, t);
		const sectors = `${KERNEL_NAMES[k]}: ${t.loadSectors + t.storeSectors} sectors`;
		const passes = k === 'naive' ? '' : `, ${plural(t.sharedPasses, 'shared memory pass', 'shared memory passes')}`;
		const est = e.gbPerS === null ? ', nothing to move' : `, model estimate ${fmt(e.gbPerS)} GB/s of ${fmt(g.memoryGbPerS)} on the ${g.name}`;
		return `${sectors}${passes}${est}.`;
	}).join(' ');
}

/** Source lines (1-based) shown in the sim's code panel, per step. */
export const CODE: Record<KernelId, string> = {
	naive: `int x = blockIdx.x * TILE + threadIdx.x;
int y = blockIdx.y * TILE + threadIdx.y;
for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < n && y + j < n)
        out[x * n + (y + j)] = in[(y + j) * n + x];`,
	tiled: `__shared__ float tile[TILE][TILE];
int x = blockIdx.x * TILE + threadIdx.x;
int y = blockIdx.y * TILE + threadIdx.y;
for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < n && y + j < n)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
__syncthreads();
x = blockIdx.y * TILE + threadIdx.x;
y = blockIdx.x * TILE + threadIdx.y;
for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < n && y + j < n)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];`,
	padded: '',
};
CODE.padded = CODE.tiled.replace('tile[TILE][TILE]', 'tile[TILE][TILE + 1]');

export function lineAt(k: KernelId, step: number): number | null {
	if (k === 'naive') return step === 1 || step === 2 ? 5 : null;
	if (step === 1 || step === 2) return 6;
	if (step === 3) return 7;
	if (step === 4 || step === 5) return 12;
	return null;
}
