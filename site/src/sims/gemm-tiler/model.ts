import { gpuById, type GpuSpec } from '@sim-core/gpu-specs';
import { clockGhz } from '../transpose/model';

/**
 * The GEMM kernels of chapter 20, C = A * B with square M = N = K float32
 * matrices in row-major order, counted per element of C.
 *
 * Global memory: every requested 32-byte sector is charged as DRAM traffic.
 * L1 and L2 are left out on purpose, so the counts show what each kernel asks
 * for, not what the caches save it.
 * Shared memory: one SM serves one pass per clock, up to 32 distinct 4-byte
 * words in 32 different banks (chapter 8). A warp's load costs as many passes
 * as the most distinct words any bank must serve, and at least distinct words / 32,
 * which also covers 16-byte loads. The passes of warp 0 at k = 0 stand for
 * every warp and every k. Tile stores into shared memory are counted as
 * conflict-free.
 * Compute: each preset's peak FP32 rate.
 */
export const STEP_IDS = ['naive', 'coalesced', 'smem', 'reg1d', 'reg2d', 'vec', 'autotune'] as const;
export type StepId = (typeof STEP_IDS)[number];
export const STEP_NAMES: Record<StepId, string> = {
	naive: 'Naive',
	coalesced: 'Coalesced',
	smem: 'Shared memory tiling',
	reg1d: '1D register blocking',
	reg2d: '2D register blocking',
	vec: 'Vectorized float4 loads',
	autotune: 'Autotuned tiles',
};
export const LAST_STEP = STEP_IDS.length - 1;
export const stepAt = (step: number): StepId => STEP_IDS[Math.min(LAST_STEP, Math.max(0, step))];

export const WARP = 32;
export const SECTOR = 32;
export const BANKS = 32;
export const SIZES = [1024, 2048, 4096] as const;
/** Naive and coalesced kernels launch 32 x 32 threads per block. */
export const NAIVE_BLOCK = 32;
/** Static __shared__ arrays are limited to 48 KB per block; more needs dynamic shared memory and an opt-in. */
export const STATIC_SMEM_LIMIT = 48 * 1024;

export interface Tile {
	bm: number;
	bn: number;
	bk: number;
	tm: number;
	tn: number;
}
export type Param = keyof Tile;
export const PARAMS: Param[] = ['bm', 'bn', 'bk', 'tm', 'tn'];
/** Allowed values as powers of two: [min, max] exponents. */
export const RANGE: Record<Param, [number, number]> = { bm: [4, 8], bn: [4, 8], bk: [2, 6], tm: [0, 4], tn: [0, 4] };

/** Which tile sizes each step lets the reader change. */
export const EDITABLE: Record<StepId, Param[]> = {
	naive: [],
	coalesced: [],
	smem: ['bm', 'bn', 'bk'],
	reg1d: ['bm', 'bn', 'bk', 'tm'],
	reg2d: PARAMS,
	vec: PARAMS,
	autotune: [],
};

/** The tile sizes of Boehm's worklog for each kernel. */
export const PRESETS: Record<StepId, Tile> = {
	naive: { bm: 32, bn: 32, bk: 1, tm: 1, tn: 1 },
	coalesced: { bm: 32, bn: 32, bk: 1, tm: 1, tn: 1 },
	smem: { bm: 32, bn: 32, bk: 32, tm: 1, tn: 1 },
	reg1d: { bm: 64, bn: 64, bk: 8, tm: 8, tn: 1 },
	reg2d: { bm: 128, bn: 128, bk: 8, tm: 8, tn: 8 },
	vec: { bm: 128, bn: 128, bk: 8, tm: 8, tn: 8 },
	autotune: { bm: 128, bn: 128, bk: 8, tm: 8, tn: 8 },
};
export const presetTiles = (): Record<StepId, Tile> =>
	Object.fromEntries(STEP_IDS.map((s) => [s, { ...PRESETS[s] }])) as Record<StepId, Tile>;

export interface Config {
	gpu: GpuSpec['id'];
	/** M = N = K. */
	size: number;
	tiles: Record<StepId, Tile>;
}

/** The tile a step actually runs with: fixed fields forced, the autotuned step searched. */
export function effectiveTile(c: Config, s: StepId): Tile {
	if (s === 'autotune') return autotune(gpuById(c.gpu), c.size).best?.tile ?? PRESETS.vec;
	const t = { ...c.tiles[s] };
	if (s === 'naive' || s === 'coalesced') return { ...PRESETS[s] };
	if (s === 'smem') t.tm = t.tn = 1;
	if (s === 'reg1d') t.tn = 1;
	return t;
}

/** The tile layout the step's kernel uses; autotune runs the vectorized kernel. */
const kernelOf = (s: StepId): StepId => (s === 'autotune' ? 'vec' : s);

const sectorsPerRow = (floats: number) => Math.max(1, Math.ceil((floats * 4) / SECTOR));

/** Passes one warp-wide shared load needs; `words` lists the words each active lane reads. */
export function passesOf(words: number[][]): number {
	const distinct = new Set(words.flat());
	const perBank = new Array(BANKS).fill(0);
	for (const w of distinct) perBank[w % BANKS]++;
	return Math.max(Math.ceil(distinct.size / BANKS), ...perBank);
}

const range = (n: number, step = 1) => Array.from({ length: Math.ceil(n / step) }, (_, i) => i * step);

/** The shared loads warp 0 issues for one k of the inner loop, as word lists per lane. */
export function warpSharedLoads(s: StepId, t: Tile): { what: 'A' | 'B'; words: number[][] }[] {
	const k = kernelOf(s);
	if (k === 'naive' || k === 'coalesced') return [];
	const cols = t.bn / t.tn;
	const lanes = range(Math.min(WARP, threads(s, t)));
	const tr = (l: number) => Math.floor(l / cols);
	const tc = (l: number) => l % cols;
	if (k === 'vec') {
		// A's tile is stored transposed, AsT[k * BM + m]; each lane reads 4 words per instruction.
		const quad = (w: number) => [w, w + 1, w + 2, w + 3];
		return [
			...range(t.tm, 4).map((i) => ({ what: 'A' as const, words: lanes.map((l) => quad(tr(l) * t.tm + i)) })),
			...range(t.tn, 4).map((j) => ({ what: 'B' as const, words: lanes.map((l) => quad(tc(l) * t.tn + j)) })),
		];
	}
	// As[m * BK + k] and Bs[k * BN + n], at k = 0.
	return [
		...range(t.tm).map((i) => ({ what: 'A' as const, words: lanes.map((l) => [(tr(l) * t.tm + i) * t.bk]) })),
		...range(t.tn).map((j) => ({ what: 'B' as const, words: lanes.map((l) => [tc(l) * t.tn + j]) })),
	];
}

export function threads(s: StepId, t: Tile) {
	const k = kernelOf(s);
	if (k === 'naive' || k === 'coalesced') return NAIVE_BLOCK * NAIVE_BLOCK;
	return (t.bm / t.tm) * (t.bn / t.tn);
}

export interface Resources {
	threads: number;
	sharedBytes: number;
	/** Rough: accumulators + register copies of A and B + 24 for indices, pointers and loop counters. */
	regsPerThread: number;
	blocksPerSm: number;
	/** What limits blocksPerSm. */
	blockLimit: 'blocks' | 'warps' | 'registers' | 'shared memory';
	/** Configs that cannot run as written. */
	errors: string[];
	/** Configs that run but deserve a note. */
	notes: string[];
}

export function resources(g: GpuSpec, s: StepId, t: Tile): Resources {
	const k = kernelOf(s);
	const n = threads(s, t);
	const tiled = k !== 'naive' && k !== 'coalesced';
	const sharedBytes = tiled ? 4 * (t.bm * t.bk + t.bk * t.bn) : 0;
	const regsPerThread = 24 + t.tm * t.tn + (k === 'smem' ? 0 : t.tm + t.tn);
	const errors: string[] = [];
	const notes: string[] = [];
	if (t.tm > t.bm || t.tn > t.bn) errors.push('TM and TN must not exceed BM and BN');
	if (n > g.maxThreadsPerBlock) errors.push(`${fmt(n)} threads per block; the limit is ${fmt(g.maxThreadsPerBlock)}`);
	if (sharedBytes > g.maxSharedKbPerBlock * 1024) {
		errors.push(`${kb(sharedBytes)} of shared memory per block; the ${g.name} allows ${g.maxSharedKbPerBlock} KB`);
	} else if (sharedBytes > STATIC_SMEM_LIMIT) {
		errors.push(`${kb(sharedBytes)} of static shared memory; the limit is 48 KB without dynamic shared memory and an opt-in`);
	}
	if (regsPerThread > g.maxRegsPerThread) errors.push(`about ${regsPerThread} registers per thread; above ${g.maxRegsPerThread} the compiler spills to local memory`);
	if (k === 'vec' && [t.tm, t.tn, t.bk, t.bn].some((v) => v % 4 !== 0)) errors.push('float4 loads need TM, TN, BK and BN to be multiples of 4');
	if (n < WARP && errors.length === 0) notes.push(`${n} threads per block: ${WARP - n} lanes of the only warp have no work`);

	const warps = Math.ceil(n / WARP);
	const regsPerWarp = Math.ceil((regsPerThread * WARP) / g.regAllocUnit) * g.regAllocUnit;
	const smem = sharedBytes ? Math.ceil(sharedBytes / g.smemAllocUnit) * g.smemAllocUnit + g.reservedSmemPerBlock : 0;
	const limits: [Resources['blockLimit'], number][] = [
		['blocks', g.maxBlocksPerSm],
		['warps', Math.floor(g.maxWarpsPerSm / warps)],
		['registers', Math.floor((g.registerFileKbPerSm * 256) / (regsPerWarp * warps))],
		['shared memory', smem ? Math.floor((g.maxSharedKbPerSm * 1024) / smem) : Infinity],
	];
	const [blockLimit, blocksPerSm] = limits.reduce((a, b) => (b[1] < a[1] ? b : a));
	if (regsPerWarp * warps > g.maxRegsPerBlock && regsPerThread <= g.maxRegsPerThread) {
		errors.push(`about ${fmt(regsPerWarp * warps)} registers per block; the limit is ${fmt(g.maxRegsPerBlock)}`);
	}
	if (errors.length === 0 && blocksPerSm < 1) errors.push('not even one block fits on an SM');
	return { threads: n, sharedBytes, regsPerThread, blocksPerSm: Math.max(0, blocksPerSm), blockLimit, errors, notes };
}

export type Limit = 'compute' | 'dram' | 'shared';

export interface Analysis {
	step: StepId;
	tile: Tile;
	flopsPerOutput: number;
	/** Floats the kernel loads from global memory per element of C. */
	globalFloats: number;
	/** Bytes of 32-byte sectors moved per element of C, including its store. */
	dramBytes: number;
	/** Floats a thread reads from shared memory per element of C it computes. */
	sharedFloats: number;
	/** Shared memory passes per element of C, loads and tile stores. */
	sharedPasses: number;
	/** FLOPs per byte of DRAM sector traffic. */
	intensity: number;
	/** How many times each element of A and of B is loaded from global memory over the whole multiply. */
	reuse: { a: number; b: number };
	/** Model ceilings in GFLOP/s. */
	roofs: Record<Limit, number>;
	/** Model estimate in GFLOP/s, the lowest roof; null when the config cannot run. */
	gflops: number | null;
	limit: Limit;
	res: Resources;
}

export const peakGflops = (g: GpuSpec) => g.peakFp32Tflops * 1000;

export function analyze(g: GpuSpec, size: number, s: StepId, t: Tile): Analysis {
	const K = size;
	const k = kernelOf(s);
	const flopsPerOutput = 2 * K;
	let globalFloats: number;
	let dramBytes: number;
	let sharedFloats = 0;
	let sharedPasses = 0;
	let reuse: Analysis['reuse'];
	if (k === 'naive') {
		// Lanes walk rows: A is 32 rows apart (32 sectors per warp), B is one word for all lanes (1 sector).
		globalFloats = 2 * K;
		dramBytes = ((32 + 1) * SECTOR * K) / WARP + SECTOR;
		reuse = { a: size, b: size };
	} else if (k === 'coalesced') {
		// Lanes walk columns: A is one word for all lanes (1 sector), B is 32 neighbours (4 sectors).
		globalFloats = 2 * K;
		dramBytes = ((1 + 4) * SECTOR * K) / WARP + 4;
		reuse = { a: size, b: size };
	} else {
		globalFloats = K * (1 / t.bn + 1 / t.bm);
		const perTile = (t.bm * sectorsPerRow(t.bk) + t.bk * sectorsPerRow(t.bn)) * SECTOR;
		dramBytes = ((K / t.bk) * perTile) / (t.bm * t.bn) + 4;
		sharedFloats = (K * (t.tm + t.tn)) / (t.tm * t.tn);
		const lanes = Math.min(WARP, threads(s, t));
		const loadPasses = warpSharedLoads(s, t).reduce((sum, l) => sum + passesOf(l.words), 0);
		sharedPasses = K * (loadPasses / (lanes * t.tm * t.tn) + (t.bm + t.bn) / (BANKS * t.bm * t.bn));
		reuse = { a: Math.ceil(size / t.bn), b: Math.ceil(size / t.bm) };
	}
	const roofs: Record<Limit, number> = {
		compute: peakGflops(g),
		dram: (flopsPerOutput / dramBytes) * g.memoryGbPerS,
		shared: sharedPasses ? (flopsPerOutput * g.sms * clockGhz(g)) / sharedPasses : Infinity,
	};
	const limit = (Object.keys(roofs) as Limit[]).reduce((a, b) => (roofs[b] < roofs[a] ? b : a));
	const res = resources(g, s, t);
	return {
		step: s,
		tile: t,
		flopsPerOutput,
		globalFloats,
		dramBytes,
		sharedFloats,
		sharedPasses,
		intensity: flopsPerOutput / dramBytes,
		reuse,
		roofs,
		gflops: res.errors.length ? null : roofs[limit],
		limit,
		res,
	};
}

export const analyzeStep = (c: Config, s: StepId) => analyze(gpuById(c.gpu), c.size, s, effectiveTile(c, s));

export const SEARCH = { bm: [32, 64, 128, 256], bn: [32, 64, 128, 256], bk: [8, 16, 32], tm: [4, 8, 16], tn: [4, 8, 16] };

export interface Search {
	searched: number;
	valid: number;
	/** Valid configs within 1% of the best estimate. */
	tied: number;
	best: Analysis | null;
	top: Analysis[];
}

const cache = new Map<string, Search>();

/**
 * Model-driven search over the vectorized kernel's tile sizes. Ranks by the
 * estimate, then by fewer shared passes, then by more blocks per SM. A real
 * autotuner times each candidate instead.
 */
export function autotune(g: GpuSpec, size: number): Search {
	const key = `${g.id}:${size}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const all: Analysis[] = [];
	for (const bm of SEARCH.bm)
		for (const bn of SEARCH.bn)
			for (const bk of SEARCH.bk)
				for (const tm of SEARCH.tm)
					for (const tn of SEARCH.tn) all.push(analyze(g, size, 'vec', { bm, bn, bk, tm, tn }));
	const ok = all.filter((a) => a.gflops !== null);
	ok.sort((x, y) => y.gflops! - x.gflops! || x.sharedPasses - y.sharedPasses || y.res.blocksPerSm - x.res.blocksPerSm);
	const best = ok[0] ?? null;
	const result: Search = {
		searched: all.length,
		valid: ok.length,
		tied: best ? ok.filter((a) => a.gflops! >= best.gflops! * 0.99).length : 0,
		best: best && { ...best, step: 'autotune' },
		top: ok.slice(0, 3),
	};
	cache.set(key, result);
	return result;
}

export const fmt = (n: number, digits = 0) =>
	n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
/** Three significant figures for small per-output counts. */
export const num = (n: number) => (n >= 100 ? fmt(n) : n >= 1 ? fmt(n, n >= 10 ? 1 : 2) : n.toPrecision(2));
export const kb = (bytes: number) => `${fmt(bytes / 1024, bytes % 1024 ? 1 : 0)} KB`;
export const tileText = (s: StepId, t: Tile) => {
	if (s === 'naive' || s === 'coalesced') return `${NAIVE_BLOCK} x ${NAIVE_BLOCK} threads per block`;
	const base = `BM=${t.bm} BN=${t.bn} BK=${t.bk}`;
	if (s === 'smem') return base;
	if (s === 'reg1d') return `${base} TM=${t.tm}`;
	return `${base} TM=${t.tm} TN=${t.tn}`;
};
export const LIMIT_NAMES: Record<Limit, string> = { compute: 'FP32 compute', dram: 'DRAM bandwidth', shared: 'shared memory passes' };

const WHY: Record<StepId, string> = {
	naive: 'Lanes of a warp walk down a column of C, so each load of A touches 32 sectors for 32 useful words.',
	coalesced: 'Lanes walk along a row of C: A is one broadcast word, B is 32 neighbouring words in 4 sectors.',
	smem: 'Each tile of A and B is read from global memory once per block, then reused from shared memory.',
	reg1d: 'A value of B in a register feeds TM multiply-adds, so each output needs fewer shared loads.',
	reg2d: 'TM + TN shared loads feed TM x TN multiply-adds per k.',
	vec: "16-byte loads, and A's tile stored transposed, cut the shared memory passes per output.",
	autotune: 'The search tries every tile shape in the grid that fits the GPU and keeps the best model estimate.',
};

export function describe(c: Config, step: number): string {
	const s = stepAt(step);
	const g = gpuById(c.gpu);
	const a = analyzeStep(c, s);
	const head = `Step ${step + 1} of ${LAST_STEP + 1}, ${STEP_NAMES[s]} (${tileText(s, a.tile)}). ${WHY[s]}`;
	if (a.gflops === null) return `${head} This config cannot run: ${a.res.errors.join('; ')}.`;
	const nums = ` Per output: ${num(a.globalFloats)} floats from global memory, ${num(a.dramBytes)} bytes of sectors, ${num(a.sharedPasses)} shared passes; ${num(a.intensity)} FLOPs per byte.`;
	const est = ` Model estimate ${fmt(a.gflops)} GFLOP/s on the ${g.name}, limited by ${LIMIT_NAMES[a.limit]}.`;
	if (s !== 'autotune') return head + nums + est;
	const r = autotune(g, c.size);
	return `${head} ${r.valid} of ${r.searched} configs fit, ${r.tied} within 1% of the best.${nums}${est}`;
}

/** Inner loops shown in the code panel; the highlighted line is what the step changed. */
export const CODE: Record<StepId, string> = {
	naive: `int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;`,
	coalesced: `int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;`,
	smem: `for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile<BM, BK>(As, A, row0, k0);
    load_tile<BK, BN>(Bs, B, k0, col0);
    __syncthreads();
    for (int k = 0; k < BK; ++k)
        acc += As[tr * BK + k] * Bs[k * BN + tc];
    __syncthreads();
}`,
	reg1d: `for (int k = 0; k < BK; ++k) {
    float b = Bs[k * BN + tc];
    for (int i = 0; i < TM; ++i)
        acc[i] += As[(tr * TM + i) * BK + k] * b;
}`,
	reg2d: `for (int k = 0; k < BK; ++k) {
    for (int i = 0; i < TM; ++i) a[i] = As[(tr * TM + i) * BK + k];
    for (int j = 0; j < TN; ++j) b[j] = Bs[k * BN + tc * TN + j];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] += a[i] * b[j];
}`,
	vec: `for (int k = 0; k < BK; ++k) {
    for (int i = 0; i < TM; i += 4)
        load4(&a[i], &AsT[k * BM + tr * TM + i]);
    for (int j = 0; j < TN; j += 4)
        load4(&b[j], &Bs[k * BN + tc * TN + j]);
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] += a[i] * b[j];
}`,
	autotune: `for (auto &cfg : candidates) {
    float ms = time_ms(cfg.launch);
    if (ms < best_ms) best = cfg, best_ms = ms;
}`,
};

export const LINES: Record<StepId, number[]> = {
	naive: [1, 5],
	coalesced: [1, 5],
	smem: [2, 3, 6],
	reg1d: [2, 4],
	reg2d: [2, 3, 6],
	vec: [3, 5],
	autotune: [2, 3],
};
