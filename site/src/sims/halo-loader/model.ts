/**
 * One block of a tiled 2D convolution with a (2R+1) x (2R+1) filter. The
 * block has T x T threads and computes a T x T output tile. Every output
 * needs the input pixels within R of it, so the block copies a
 * (T + 2R) x (T + 2R) input tile into shared memory: its own T x T pixels
 * plus a halo R pixels wide on every side.
 *
 * The T^2 threads fill the (T + 2R)^2 slots in rounds, thread t taking slots
 * t, t + T^2, t + 2 T^2, ..., the loop in kernels/ch16-stencil. Pixels outside
 * the image are zero (zero padding): the kernel writes 0 to their slots and
 * issues no load for them.
 *
 * Loads are counted as the element loads the kernel issues. The L1 and L2
 * caches (chapter 9) serve many of the naive kernel's repeated loads; the
 * model leaves them out.
 */
export type Position = 'inside' | 'corner';
export const POSITIONS: Position[] = ['inside', 'corner'];
export const TILE_RANGE = { min: 4, max: 32, step: 4 };
export const RADIUS_RANGE = { min: 1, max: 8, step: 1 };

/**
 * __constant__ memory holds 64 KB per device on every compute capability.
 * CUDA Programming Guide, "Compute Capabilities", technical specifications table:
 * https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
 */
export const CONSTANT_MEMORY_BYTES = 64 * 1024;

export interface Config {
	tile: number;
	radius: number;
	position: Position;
}

export const side = (c: Config) => c.tile + 2 * c.radius;
export const width = (c: Config) => 2 * c.radius + 1;
export const slots = (c: Config) => side(c) ** 2;
export const threads = (c: Config) => c.tile ** 2;
export const rounds = (c: Config) => Math.ceil(slots(c) / threads(c));
export const haloCells = (c: Config) => slots(c) - threads(c);
/** Halo cells per output cell of the tile. */
export const overhead = (c: Config) => haloCells(c) / threads(c);
/** Bytes of shared memory for the float tile, and of constant memory for the float filter. */
export const sharedBytes = (c: Config) => 4 * slots(c);
export const filterBytes = (c: Config) => 4 * width(c) ** 2;

/**
 * Whether slot (r, q) of the input tile is inside the image. 'corner' puts the
 * block at the image's top-left corner, so the first R rows and columns of the
 * tile lie above and left of the image. 'inside' is any block at least R pixels
 * from every edge.
 */
export const inImage = (c: Config, r: number, q: number) => c.position === 'inside' || (r >= c.radius && q >= c.radius);
export const isHalo = (c: Config, r: number, q: number) =>
	r < c.radius || q < c.radius || r >= c.radius + c.tile || q >= c.radius + c.tile;

/** How many of the block's T x T outputs have slot (r, q) in their window: 1D overlap of [r - 2R, r] with [0, T - 1], squared out. */
export function readers(c: Config, r: number, q: number) {
	const span = (x: number) => Math.min(x, c.tile - 1) - Math.max(0, x - 2 * c.radius) + 1;
	return span(r) * span(q);
}

/** The round in which a slot is filled, 0-based. */
export const roundOf = (c: Config, r: number, q: number) => Math.floor((r * side(c) + q) / threads(c));

export interface Totals {
	/** Global loads the tiled kernel issues: the in-image slots. */
	tiled: number;
	/** Global loads the naive kernel issues: every in-image pixel of every output's window. */
	naive: number;
	/** Shared memory reads in the compute loop: T^2 (2R+1)^2. */
	sharedReads: number;
	/** Slots filled with 0 because they lie outside the image. */
	zeros: number;
	tiledPerOutput: number;
	naivePerOutput: number;
	/** Fraction of the tiled kernel's loads that are halo pixels. */
	haloShare: number;
}

export function totals(c: Config): Totals {
	const s = side(c);
	let tiled = 0;
	let naive = 0;
	let reads = 0;
	let halo = 0;
	for (let r = 0; r < s; r++)
		for (let q = 0; q < s; q++) {
			const n = readers(c, r, q);
			reads += n;
			if (!inImage(c, r, q)) continue;
			tiled++;
			naive += n;
			if (isHalo(c, r, q)) halo++;
		}
	const t = threads(c);
	return {
		tiled,
		naive,
		sharedReads: reads,
		zeros: slots(c) - tiled,
		tiledPerOutput: tiled / t,
		naivePerOutput: naive / t,
		haloShare: halo / tiled,
	};
}

export type Kind = 'start' | 'load' | 'sync' | 'compute' | 'store';
export interface Step {
	kind: Kind;
	/** Load round, or -1. */
	round: number;
	/** 1-based lines of CODE to highlight. */
	lines: number[];
}

export const CODE = `__constant__ float c_filter[(2 * R + 1) * (2 * R + 1)];

__global__ void conv2d_tiled(const float *in, float *out, int h, int w) {
    constexpr int IN = TILE + 2 * R;
    __shared__ float tile[IN][IN];
    int x0 = blockIdx.x * TILE - R, y0 = blockIdx.y * TILE - R;
    for (int i = threadIdx.y * TILE + threadIdx.x; i < IN * IN; i += TILE * TILE) {
        int yy = y0 + i / IN, xx = x0 + i % IN;
        bool inside = yy >= 0 && yy < h && xx >= 0 && xx < w;
        tile[i / IN][i % IN] = inside ? in[yy * w + xx] : 0.0f;
    }
    __syncthreads();
    int x = x0 + R + threadIdx.x, y = y0 + R + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int dy = 0; dy <= 2 * R; ++dy)
        for (int dx = 0; dx <= 2 * R; ++dx)
            sum += c_filter[dy * (2 * R + 1) + dx] * tile[threadIdx.y + dy][threadIdx.x + dx];
    out[y * w + x] = sum;
}`;

export function steps(c: Config): Step[] {
	const out: Step[] = [{ kind: 'start', round: -1, lines: [4, 5, 6] }];
	for (let k = 0; k < rounds(c); k++) out.push({ kind: 'load', round: k, lines: [7, 8, 9, 10] });
	out.push({ kind: 'sync', round: -1, lines: [12] });
	out.push({ kind: 'compute', round: -1, lines: [13, 14, 15, 16, 17, 18] });
	out.push({ kind: 'store', round: -1, lines: [19] });
	return out;
}

export const lastStep = (c: Config) => rounds(c) + 3;

export interface State {
	step: number;
	cur: Step;
	/** Rounds finished, including the current one. */
	loadedRounds: number;
	/** Global loads and zero fills so far. */
	loads: number;
	zeros: number;
	/** True once the compute loop has run. */
	computed: boolean;
	stored: boolean;
}

export function stateAt(c: Config, step: number): State {
	const all = steps(c);
	const s = Math.max(0, Math.min(step, all.length - 1));
	const cur = all[s];
	const loadedRounds = cur.kind === 'start' ? 0 : cur.kind === 'load' ? cur.round + 1 : rounds(c);
	const filled = Math.min(slots(c), loadedRounds * threads(c));
	let loads = 0;
	const sd = side(c);
	for (let i = 0; i < filled; i++) if (inImage(c, Math.floor(i / sd), i % sd)) loads++;
	return {
		step: s,
		cur,
		loadedRounds,
		loads,
		zeros: filled - loads,
		computed: cur.kind === 'compute' || cur.kind === 'store',
		stored: cur.kind === 'store',
	};
}

const fmt = (x: number) => x.toLocaleString('en-US');
export const fix = (x: number, digits = 2) => (Number.isInteger(x) ? fmt(x) : x.toFixed(digits));
export const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

/** Counts for one load round. */
export function roundCounts(c: Config, k: number) {
	const sd = side(c);
	const first = k * threads(c);
	const end = Math.min(slots(c), first + threads(c));
	let loads = 0;
	let halo = 0;
	for (let i = first; i < end; i++) {
		const r = Math.floor(i / sd);
		const q = i % sd;
		if (!inImage(c, r, q)) continue;
		loads++;
		if (isHalo(c, r, q)) halo++;
	}
	return { first, end, active: end - first, loads, halo, zeros: end - first - loads };
}

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	const T = c.tile;
	const R = c.radius;
	const k = width(c);
	const sd = side(c);
	const t = totals(c);
	const x = st.cur;
	switch (x.kind) {
		case 'start':
			return (
				`A block of ${T} x ${T} = ${fmt(threads(c))} threads computes one ${T} x ${T} output tile with a ${k} x ${k} filter. ` +
				`Each output needs the pixels within ${R} of it, so the block needs a ${sd} x ${sd} input tile: its own ${fmt(threads(c))} pixels plus a halo of ${fmt(haloCells(c))}. ` +
				`Predict: how many global loads per output does the tiled kernel issue, and how many does the naive one? Then press Next.`
			);
		case 'load': {
			const rc = roundCounts(c, x.round);
			const who =
				rc.active === threads(c)
					? `every thread fills one slot, i = ${fmt(rc.first)} to ${fmt(rc.end - 1)}`
					: `threads 0 to ${fmt(rc.active - 1)} fill slots ${fmt(rc.first)} to ${fmt(rc.end - 1)}; the other ${fmt(threads(c) - rc.active)} have nothing left to load`;
			const zeros = rc.zeros ? ` ${fmt(rc.zeros)} slots lie outside the image and get 0 with no load.` : '';
			return (
				`Round ${x.round + 1} of ${rounds(c)}: ${who}. ${fmt(rc.loads)} global loads, ${fmt(rc.halo)} of them halo pixels.${zeros} ` +
				`Loaded so far: ${fmt(st.loads)} of ${fmt(t.tiled)}.`
			);
		}
		case 'sync':
			return (
				`__syncthreads(). The tile is full: ${fmt(t.tiled)} pixels loaded, each exactly once` +
				(t.zeros ? `, and ${fmt(t.zeros)} zeros for the padding` : '') +
				`. Many slots a thread needs were filled by other threads, so no thread may read the tile before every thread has written its part.`
			);
		case 'compute':
			return (
				`Every thread reads its ${k} x ${k} window from shared memory and each weight from constant memory, where all threads of a warp read the same weight at once. ` +
				`The outline is thread (0, 0)'s window, reaching ${R} into the halo. Numbers count the threads reading each slot: ${fmt(t.sharedReads)} shared reads in all. ` +
				`The naive kernel issues ${fmt(t.naive)} global loads for the same windows.`
			);
		case 'store':
			return (
				`Done: ${fmt(t.tiled)} global loads for ${fmt(threads(c))} outputs, ${fix(t.tiledPerOutput)} per output, against ${fix(t.naivePerOutput)} for the naive kernel, ` +
				`${fix(t.naive / t.tiled, 1)} times fewer. Halo pixels are ${pct(t.haloShare)} of the loads.`
			);
	}
}
