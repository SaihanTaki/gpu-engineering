/**
 * Softmax of one row, y[i] = exp(x[i]) / sum_j exp(x[j]), computed three ways,
 * chunk by chunk, as in Milakov and Gimelshein, "Online normalizer calculation
 * for softmax" (arXiv:1805.02867): naive (their Algorithm 1), safe with a
 * separate max pass (Algorithm 2), and online (Algorithm 3), where a running
 * max m and sum d are kept and d is rescaled by exp(m_old - m_new) when the max
 * grows. Processing a chunk is the paper's merge of two [m, d] pairs (its
 * Section 3.1): the running pair and the chunk's own pair.
 *
 * Arithmetic is done in double precision. Only the range of the chosen format
 * is modeled, not its rounding: an exp() or partial sum above the largest
 * finite value becomes Infinity and one below half the smallest subnormal
 * becomes 0, as in IEEE 754 arithmetic.
 */
export const N = 16;
export const CHUNKS = [1, 2, 4, 8, 16] as const;

export const MODE_IDS = ['naive', 'safe', 'online'] as const;
export type Mode = (typeof MODE_IDS)[number];
export const MODE_NAMES: Record<Mode, string> = {
	naive: 'Naive: sum, then normalize',
	safe: 'Safe 3-pass: max, sum, normalize',
	online: 'Online: max and sum together, then normalize',
};

export const PRESET_IDS = ['random', 'increasing', 'huge'] as const;
export type Preset = (typeof PRESET_IDS)[number];
export const PRESET_NAMES: Record<Preset, string> = {
	random: 'Random, -4 to 4',
	increasing: 'Increasing, 0 to 12',
	huge: 'Random with one 100',
};

export const FORMAT_IDS = ['fp32', 'fp16'] as const;
export type Format = (typeof FORMAT_IDS)[number];
/** Largest finite value of each IEEE 754 format, and ln of it: exp(x) overflows above that x. */
export const MAX_FINITE: Record<Format, number> = { fp32: 3.4028234663852886e38, fp16: 65504 };
/** Smallest positive subnormal of each format. */
export const MIN_SUBNORMAL: Record<Format, number> = { fp32: 2 ** -149, fp16: 2 ** -24 };
export const FORMAT_NAMES: Record<Format, string> = { fp32: 'FP32', fp16: 'FP16' };
export const expLimit = (f: Format) => Math.log(MAX_FINITE[f]);

export interface Config {
	mode: Mode;
	preset: Preset;
	chunk: number;
	format: Format;
}

/** Values with one decimal so every cell is readable; a fixed seed so every reader sees the same row. */
export function inputs(p: Preset): number[] {
	let s = 7;
	const rand = () => {
		s = (s * 1103515245 + 12345) % 2147483648;
		return s / 2147483648;
	};
	const random = Array.from({ length: N }, () => Math.round((rand() * 8 - 4) * 10) / 10);
	if (p === 'random') return random;
	if (p === 'increasing') return Array.from({ length: N }, (_, i) => Math.round(i * 0.8 * 10) / 10);
	const out = [...random];
	out[11] = 100;
	return out;
}

/** The exact softmax, by the safe method in double precision. */
export function exact(x: number[]): number[] {
	const m = Math.max(...x);
	const e = x.map((v) => Math.exp(v - m));
	const d = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / d);
}

/** One merge of the paper's operator: [m1, d1] (+) [m2, d2]. */
export function mergePair(a: { m: number; d: number }, b: { m: number; d: number }) {
	const m = Math.max(a.m, b.m);
	if (m === -Infinity) return { m, d: 0 };
	return { m, d: a.d * Math.exp(a.m - m) + b.d * Math.exp(b.m - m) };
}

export type PassKind = 'max' | 'sum' | 'stats' | 'normalize';
export const PASSES: Record<Mode, PassKind[]> = {
	naive: ['sum', 'normalize'],
	safe: ['max', 'sum', 'normalize'],
	online: ['stats', 'normalize'],
};

export interface Step {
	kind: 'start' | PassKind | 'done';
	/** Index into PASSES[mode], -1 for start and done. */
	pass: number;
	/** Element range [lo, hi) this step reads. */
	lo: number;
	hi: number;
	/** Running state after the step; m is -Infinity until a max is known. */
	m: number;
	d: number;
	/** Online only: the max before this chunk, the chunk's own max and sum, and exp(m_old - m_new). */
	mOld: number | null;
	chunkM: number | null;
	chunkD: number | null;
	rescale: number | null;
	/** Outputs written so far, null where not yet written. */
	y: (number | null)[];
	/** Indices whose exp() overflowed to Infinity in this format, over the whole run so far. */
	overflow: number[];
	/** Elements read and written so far. */
	reads: number;
	writes: number;
	lines: number[];
}

export const CODE: Record<Mode, string> = {
	naive: `float d = 0.0f;
for (int i = 0; i < n; ++i)
    d += expf(x[i]);
for (int i = 0; i < n; ++i)
    y[i] = expf(x[i]) / d;`,
	safe: `float m = -INFINITY;
for (int i = 0; i < n; ++i)
    m = fmaxf(m, x[i]);
float d = 0.0f;
for (int i = 0; i < n; ++i)
    d += expf(x[i] - m);
for (int i = 0; i < n; ++i)
    y[i] = expf(x[i] - m) / d;`,
	online: `float m = -INFINITY, d = 0.0f;
for (int c = 0; c < n; c += C) {
    float mc = -INFINITY, dc = 0.0f;
    for (int i = c; i < c + C; ++i) mc = fmaxf(mc, x[i]);
    for (int i = c; i < c + C; ++i) dc += expf(x[i] - mc);
    float m_new = fmaxf(m, mc);
    d = d * expf(m - m_new) + dc * expf(mc - m_new);
    m = m_new;
}
for (int i = 0; i < n; ++i)
    y[i] = expf(x[i] - m) / d;`,
};

const LINES: Record<Mode, Partial<Record<PassKind | 'start', number[]>>> = {
	naive: { start: [1], sum: [2, 3], normalize: [4, 5] },
	safe: { start: [1], max: [2, 3], sum: [4, 5, 6], normalize: [7, 8] },
	online: { start: [1], stats: [2, 3, 4, 5, 6, 7, 8], normalize: [10, 11] },
};

const cache = new Map<string, Step[]>();

export function steps(c: Config): Step[] {
	const key = `${c.mode}:${c.preset}:${c.chunk}:${c.format}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const x = inputs(c.preset);
	const limit = MAX_FINITE[c.format];
	const tiny = MIN_SUBNORMAL[c.format] / 2;
	const clamp = (v: number) => (v > limit ? Infinity : v < tiny ? 0 : v);
	const overflow = new Set<number>();
	const ex = (i: number, shift: number) => {
		const v = clamp(Math.exp(x[i] - shift));
		if (v === Infinity) overflow.add(i);
		return v;
	};
	let m = -Infinity;
	let d = 0;
	let reads = 0;
	let writes = 0;
	let y: (number | null)[] = x.map(() => null);
	const base = { mOld: null, chunkM: null, chunkD: null, rescale: null };
	const out: Step[] = [{ kind: 'start', pass: -1, lo: 0, hi: 0, m, d, ...base, y, overflow: [], reads, writes, lines: LINES[c.mode].start! }];

	PASSES[c.mode].forEach((kind, pass) => {
		for (let lo = 0; lo < N; lo += c.chunk) {
			const hi = Math.min(N, lo + c.chunk);
			const idx = Array.from({ length: hi - lo }, (_, k) => lo + k);
			reads += idx.length;
			let extra: Pick<Step, 'mOld' | 'chunkM' | 'chunkD' | 'rescale'> = base;
			if (kind === 'max') m = Math.max(m, ...idx.map((i) => x[i]));
			else if (kind === 'sum') {
				const shift = c.mode === 'naive' ? 0 : m;
				for (const i of idx) d = clamp(d + ex(i, shift));
			} else if (kind === 'stats') {
				const chunkM = Math.max(...idx.map((i) => x[i]));
				let chunkD = 0;
				for (const i of idx) chunkD = clamp(chunkD + ex(i, chunkM));
				const mNew = Math.max(m, chunkM);
				const rescale = clamp(Math.exp(m - mNew));
				extra = { mOld: m, chunkM, chunkD, rescale };
				d = clamp(d * rescale + chunkD * clamp(Math.exp(chunkM - mNew)));
				m = mNew;
			} else {
				const shift = c.mode === 'naive' ? 0 : m;
				y = [...y];
				for (const i of idx) y[i] = clamp(ex(i, shift) / d);
				writes += idx.length;
			}
			out.push({ kind, pass, lo, hi, m, d, ...extra, y, overflow: [...overflow].sort((a, b) => a - b), reads, writes, lines: LINES[c.mode][kind]! });
		}
	});
	const last = out[out.length - 1];
	out.push({ ...last, kind: 'done', pass: -1, lo: 0, hi: 0, ...base, lines: [] });
	cache.set(key, out);
	return out;
}

export const lastStep = (c: Config) => steps(c).length - 1;
export const stepAt = (c: Config, step: number) => {
	const all = steps(c);
	return all[Math.max(0, Math.min(step, all.length - 1))];
};

/** Largest |y - exact| over the row, NaN if any output is NaN. */
export function maxError(c: Config): number {
	const y = stepAt(c, lastStep(c)).y as number[];
	const e = exact(inputs(c.preset));
	let worst = 0;
	for (let i = 0; i < N; i++) {
		const diff = Math.abs(y[i] - e[i]);
		if (Number.isNaN(diff)) return NaN;
		worst = Math.max(worst, diff);
	}
	return worst;
}

/** Element reads and writes per row element over the whole run. */
export const accessesPerElement = (m: Mode) => ({ reads: PASSES[m].length, writes: 1 });

/** Welford's update for one value, and Chan et al.'s merge of two partial results. */
export interface Moments {
	n: number;
	mean: number;
	m2: number;
}
export const EMPTY: Moments = { n: 0, mean: 0, m2: 0 };
export function welfordAdd(a: Moments, v: number): Moments {
	const n = a.n + 1;
	const delta = v - a.mean;
	const mean = a.mean + delta / n;
	return { n, mean, m2: a.m2 + delta * (v - mean) };
}
export function welfordMerge(a: Moments, b: Moments): Moments {
	const n = a.n + b.n;
	if (n === 0) return EMPTY;
	const delta = b.mean - a.mean;
	return { n, mean: a.mean + (delta * b.n) / n, m2: a.m2 + b.m2 + (delta * delta * a.n * b.n) / n };
}
export const welford = (xs: number[]) => xs.reduce(welfordAdd, EMPTY);

/** Up to 4 significant digits; Infinity and NaN spelled out. */
export function fmt(v: number | null): string {
	if (v === null) return '';
	if (Number.isNaN(v)) return 'NaN';
	if (v === Infinity) return 'inf';
	if (v === -Infinity) return '-inf';
	if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) return v.toExponential(2).replace('e+', 'e');
	return String(Number(v.toPrecision(4)));
}

const range = (lo: number, hi: number) => (hi - lo === 1 ? `x[${lo}]` : `x[${lo}..${hi - 1}]`);

export function describe(c: Config, step: number): string {
	const s = stepAt(c, step);
	const x = inputs(c.preset);
	const passes = PASSES[c.mode].length;
	const passName = `pass ${s.pass + 1} of ${passes}`;
	const lim = expLimit(c.format).toFixed(2);
	const of = (k: number) => `${k} element${k === 1 ? '' : 's'}`;
	switch (s.kind) {
		case 'start':
			return `${MODE_NAMES[c.mode]}, ${N} values in chunks of ${c.chunk}, ${FORMAT_NAMES[c.format]}. exp(x) overflows ${FORMAT_NAMES[c.format]} for x above ${lim}; the largest input is ${fmt(Math.max(...x))}. Predict: will this method give finite probabilities, and how many times will it read the row?`;
		case 'max':
			return `${passName}, max: read ${range(s.lo, s.hi)}. Running max m = ${fmt(s.m)}. Nothing is exponentiated yet; this pass only finds the shift.`;
		case 'sum': {
			const shift = c.mode === 'naive' ? 'exp(x[i])' : `exp(x[i] - ${fmt(s.m)})`;
			const ov = s.overflow.filter((i) => i >= s.lo && i < s.hi);
			const warn = ov.length ? ` exp of ${ov.map((i) => `x[${i}] = ${fmt(x[i])}`).join(', ')} overflowed to inf, so d is inf from here on.` : '';
			return `${passName}, sum: read ${range(s.lo, s.hi)} and add ${shift}. Running sum d = ${fmt(s.d)}.${warn}`;
		}
		case 'stats': {
			const grew = s.mOld !== null && s.chunkM !== null && s.chunkM > s.mOld;
			const rescale =
				s.mOld === -Infinity
					? 'This is the first chunk, so the old sum is 0 and nothing needs rescaling.'
					: grew
						? `The max grew from ${fmt(s.mOld)} to ${fmt(s.m)}, so the old sum is multiplied by exp(${fmt(s.mOld)} - ${fmt(s.m)}) = ${fmt(s.rescale)}.`
						: `The max stays ${fmt(s.m)}, so the rescale factor is exp(0) = 1 and the chunk's sum is scaled by exp(${fmt(s.chunkM)} - ${fmt(s.m)}) = ${fmt(Math.exp(s.chunkM! - s.m))}.`;
			return `${passName}, max and sum: read ${range(s.lo, s.hi)}. Chunk max ${fmt(s.chunkM)}, chunk sum of exp(x - ${fmt(s.chunkM)}) = ${fmt(s.chunkD)}. ${rescale} Now m = ${fmt(s.m)}, d = ${fmt(s.d)}.`;
		}
		case 'normalize': {
			const ys = s.y.slice(s.lo, s.hi).map(fmt).join(', ');
			const nan = s.y.slice(s.lo, s.hi).some((v) => Number.isNaN(v)) ? ' inf / inf is NaN.' : '';
			return `${passName}, normalize: read ${range(s.lo, s.hi)} again and write y = ${ys}.${nan}`;
		}
		case 'done': {
			const err = maxError(c);
			const total = s.reads + s.writes;
			const result = Number.isNaN(err)
				? `Some outputs are NaN: the method failed in ${FORMAT_NAMES[c.format]}.`
				: `Largest difference from the exact softmax: ${err < 1e-12 ? 'below 1e-12' : fmt(err)}.`;
			return `Done. ${result} Memory: ${of(s.reads)} read and ${of(s.writes)} written, ${total / N} accesses per element. Compare the three methods below.`;
		}
	}
}
