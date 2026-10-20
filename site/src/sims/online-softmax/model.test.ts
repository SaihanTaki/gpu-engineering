import { describe as suite, expect, it } from 'vitest';
import {
	CHUNKS,
	CODE,
	describe,
	exact,
	FORMAT_IDS,
	inputs,
	lastStep,
	maxError,
	mergePair,
	MODE_IDS,
	N,
	PRESET_IDS,
	stepAt,
	steps,
	welford,
	welfordMerge,
	type Config,
} from './model';

const all: Config[] = MODE_IDS.flatMap((mode) =>
	PRESET_IDS.flatMap((preset) => CHUNKS.flatMap((chunk) => FORMAT_IDS.map((format) => ({ mode, preset, chunk, format })))),
);
const end = (c: Config) => stepAt(c, lastStep(c));

suite('online softmax', () => {
	it('inputs: 16 values, one decimal, the huge preset has one 100', () => {
		for (const p of PRESET_IDS) {
			const x = inputs(p);
			expect(x).toHaveLength(N);
			for (const v of x) expect(Math.round(v * 10) / 10).toBe(v);
		}
		expect(inputs('huge').filter((v) => v === 100)).toHaveLength(1);
		expect(Math.max(...inputs('random'))).toBeLessThanOrEqual(4);
		expect(inputs('increasing').at(-1)).toBe(12);
	});

	it('the exact softmax sums to 1', () => {
		for (const p of PRESET_IDS) expect(exact(inputs(p)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
	});

	it.each(all.filter((c) => c.mode !== 'naive'))('safe and online always match the exact result: %o', (c) => {
		expect(maxError(c)).toBeLessThan(1e-12);
		expect(end(c).overflow).toEqual([]);
	});

	it('naive overflows only where exp(max) exceeds the format', () => {
		const naive = (preset: Config['preset'], format: Config['format']) => ({ mode: 'naive', preset, chunk: 4, format }) as Config;
		expect(maxError(naive('random', 'fp32'))).toBeLessThan(1e-12);
		expect(maxError(naive('random', 'fp16'))).toBeLessThan(1e-12);
		expect(maxError(naive('increasing', 'fp32'))).toBeLessThan(1e-12);
		expect(maxError(naive('increasing', 'fp16'))).toBeNaN();
		expect(maxError(naive('huge', 'fp32'))).toBeNaN();
		// exp(100) is about 2.7e43; FP32 tops out near 3.4e38.
		expect(end(naive('huge', 'fp32')).overflow).toEqual([11]);
		// FP16 tops out at 65504, so exp(x) overflows from x = 11.1: the last two increasing values.
		expect(end(naive('increasing', 'fp16')).overflow).toEqual([14, 15]);
	});

	it.each(MODE_IDS)('memory accesses: one read per pass and one write per element (%s)', (mode) => {
		const passes = { naive: 2, safe: 3, online: 2 }[mode];
		for (const chunk of CHUNKS) {
			const c: Config = { mode, preset: 'random', chunk, format: 'fp32' };
			expect(end(c).reads).toBe(passes * N);
			expect(end(c).writes).toBe(N);
			expect(lastStep(c)).toBe(1 + passes * Math.ceil(N / chunk));
		}
	});

	it('online, chunk 1, increasing input: the max grows at every element, so every step rescales', () => {
		const c: Config = { mode: 'online', preset: 'increasing', chunk: 1, format: 'fp32' };
		const stats = steps(c).filter((s) => s.kind === 'stats');
		expect(stats[0].rescale).toBe(0);
		for (const s of stats.slice(1)) expect(s.rescale).toBeCloseTo(Math.exp(-0.8), 12);
		expect(stats.at(-1)!.m).toBe(12);
	});

	it('online, huge preset: the chunk holding the 100 rescales the old sum to almost 0', () => {
		const c: Config = { mode: 'online', preset: 'huge', chunk: 4, format: 'fp32' };
		const s = steps(c).filter((t) => t.kind === 'stats')[2];
		expect([s.lo, s.hi]).toEqual([8, 12]);
		expect(s.m).toBe(100);
		expect(s.rescale!).toBeLessThan(1e-40);
		expect(end(c).y[11]).toBeCloseTo(1, 12);
	});

	it('merging [m, d] pairs is associative and order free', () => {
		const x = inputs('random');
		const pair = (xs: number[]) => {
			const m = Math.max(...xs);
			return { m, d: xs.reduce((a, v) => a + Math.exp(v - m), 0) };
		};
		const a = pair(x.slice(0, 5));
		const b = pair(x.slice(5, 11));
		const c = pair(x.slice(11));
		const left = mergePair(mergePair(a, b), c);
		const right = mergePair(c, mergePair(b, a));
		expect(left.m).toBe(right.m);
		expect(left.d).toBeCloseTo(right.d, 12);
		expect(left.d).toBeCloseTo(pair(x).d, 12);
		expect(mergePair({ m: -Infinity, d: 0 }, a)).toEqual(a);
	});

	it('Welford and the parallel merge agree with the two-pass mean and variance', () => {
		const x = [1000.1, 1000.4, 999.8, 1000.0, 1000.3, 999.9, 1000.2];
		const mean = x.reduce((a, b) => a + b, 0) / x.length;
		const varPop = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
		const w = welford(x);
		expect(w.mean).toBeCloseTo(mean, 10);
		expect(w.m2 / w.n).toBeCloseTo(varPop, 10);
		const merged = welfordMerge(welford(x.slice(0, 3)), welford(x.slice(3)));
		expect(merged.n).toBe(7);
		expect(merged.mean).toBeCloseTo(mean, 10);
		expect(merged.m2 / merged.n).toBeCloseTo(varPop, 10);
		expect(welfordMerge(welford([]), w)).toEqual(w);
	});

	it.each(all)('every step has a status and code lines that exist: %o', (c) => {
		const code = CODE[c.mode].split('\n');
		for (let s = 0; s <= lastStep(c); s++) {
			const text = describe(c, s);
			expect(text.length).toBeGreaterThan(40);
			expect(text).not.toContain('undefined');
			for (const line of stepAt(c, s).lines) expect(code[line - 1].trim().length).toBeGreaterThan(2);
		}
	});

	it('status names the rescale and the failure', () => {
		const online: Config = { mode: 'online', preset: 'increasing', chunk: 4, format: 'fp32' };
		expect(describe(online, 2)).toContain('The max grew from 2.4 to 5.6, so the old sum is multiplied by exp(2.4 - 5.6)');
		expect(describe({ mode: 'naive', preset: 'huge', chunk: 4, format: 'fp32' }, 3)).toContain('x[11] = 100 overflowed to inf');
		expect(describe({ mode: 'naive', preset: 'huge', chunk: 16, format: 'fp32' }, 3)).toContain('Some outputs are NaN');
	});

	it('the highlighted lines are the update lines', () => {
		expect(CODE.online.split('\n')[6].trim()).toBe('d = d * expf(m - m_new) + dc * expf(mc - m_new);');
		expect(CODE.safe.split('\n')[2].trim()).toBe('m = fmaxf(m, x[i]);');
	});
});
