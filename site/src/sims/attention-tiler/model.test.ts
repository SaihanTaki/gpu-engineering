import { describe as suite, expect, it } from 'vitest';
import {
	BLOCK_OPTIONS,
	D_OPTIONS,
	describe,
	flashTraffic,
	fmtBytes,
	inputs,
	linesAt,
	N_OPTIONS,
	naiveTraffic,
	num,
	onChip,
	REAL_CONFIG,
	referenceAttention,
	simulate,
	steps,
	tileState,
	type Config,
} from './model';

const cfg = (over: Partial<Config> = {}): Config => ({ n: 12, d: 4, br: 4, bc: 4, causal: false, ...over });

const every: Config[] = [];
for (const n of N_OPTIONS)
	for (const d of D_OPTIONS)
		for (const br of BLOCK_OPTIONS)
			for (const bc of BLOCK_OPTIONS) for (const causal of [false, true]) every.push({ n, d, br, bc, causal });

suite('values', () => {
	it('the tiled online softmax gives exactly standard attention, for every setting', () => {
		for (const c of every) {
			const { q, k, v } = inputs(c.n, c.d);
			const want = referenceAttention(q, k, v, c.causal);
			const got = simulate(c).out;
			got.forEach((row, i) => row.forEach((x, e) => expect(x).toBeCloseTo(want[i][e], 12)));
		}
	});

	it('the reference itself: causal row 0 is v[0], and rows of P sum to one', () => {
		const { q, k, v } = inputs(8, 4);
		expect(referenceAttention(q, k, v, true)[0]).toEqual(v[0]);
		const ones = v.map((r) => r.map(() => 1));
		referenceAttention(q, k, ones, false).forEach((r) => r.forEach((x) => expect(x).toBeCloseTo(1, 12)));
	});

	it('inputs are one-decimal numbers in [-2, 2] that do not change with n or d', () => {
		const a = inputs(16, 8);
		const b = inputs(8, 4);
		expect(b.q[3]).toEqual(a.q[3].slice(0, 4));
		for (const x of a.k.flat()) {
			expect(Math.abs(x)).toBeLessThanOrEqual(2);
			expect(Math.round(x * 10)).toBeCloseTo(x * 10, 9);
		}
	});

	it('rescale: alpha is exp(m_old - m_new), and l follows it', () => {
		const run = simulate(cfg());
		let rescaled = 0;
		for (const f of run.frames) {
			for (const r of f.rows) {
				if (f.step.kind !== 'tile' || r.mOld === -Infinity) continue;
				expect(r.alpha).toBeCloseTo(Math.exp(r.mOld - r.mNew), 14);
				expect(r.mNew).toBeGreaterThanOrEqual(r.mOld);
				expect(r.lNew).toBeGreaterThan(r.lOld * r.alpha);
				if (r.alpha < 1) rescaled++;
			}
		}
		// The default case must show real rescales.
		expect(rescaled).toBeGreaterThan(3);
	});
});

suite('steps and tiles', () => {
	it('non-causal: every tile once, then one write per Q block', () => {
		const s = steps(cfg());
		expect(s.length).toBe(1 + 3 * 3 + 3);
		expect(s.filter((x) => x.kind === 'tile').map((x) => (x.kind === 'tile' ? `${x.i}${x.j}` : ''))).toEqual([
			'00', '01', '02', '10', '11', '12', '20', '21', '22',
		]);
	});

	it('causal with square blocks skips the tiles above the diagonal and masks the diagonal ones', () => {
		const c = cfg({ causal: true });
		const tiles = steps(c).filter((x) => x.kind === 'tile');
		expect(tiles.length).toBe(6);
		expect(tiles.filter((x) => x.kind === 'tile' && x.masked).length).toBe(3);
		const run = simulate(c);
		expect(tileState(run, 0, 1, 0)).toBe('skipped');
		expect(tileState(run, 1, 0, 0)).toBe('pending');
		expect(tileState(run, 0, 0, 1)).toBe('current');
		expect(tileState(run, 0, 0, 2)).toBe('done');
	});

	it('causal with Br=8, Bc=2, n=12: tile (0,3) holds columns 6-7, visited and masked', () => {
		const c = cfg({ br: 8, bc: 2, causal: true });
		const run = simulate(c);
		expect(run.tileStep[0].map((x) => x >= 0)).toEqual([true, true, true, true, false, false]);
		expect(run.tileStep[1].every((x) => x >= 0)).toBe(true);
	});

	it('code lines follow the step', () => {
		const s = steps(cfg({ causal: true }));
		expect(linesAt(s[0])).toEqual([]);
		expect(linesAt(s[1])).toContain(1);
		expect(linesAt(s[1])).toContain(8);
		expect(linesAt(s[s.length - 1])).toEqual([15]);
	});
});

suite('HBM traffic', () => {
	it('naive follows Algorithm 0: reads 3nd + 2n^2, writes 2n^2 + nd, FP16', () => {
		const t = naiveTraffic(cfg({ n: 16, d: 4 }));
		expect(t.read).toBe((3 * 64 + 2 * 256) * 2);
		expect(t.written).toBe((2 * 256 + 64) * 2);
		expect(t.scoreBytes).toBe(2 * 256 * 2);
		expect(naiveTraffic(cfg({ causal: true }))).toEqual(naiveTraffic(cfg()));
	});

	it('flash non-causal: Q and O once, K and V once per Q block', () => {
		for (const c of every.filter((x) => !x.causal)) {
			const tr = Math.ceil(c.n / c.br);
			const t = flashTraffic(c);
			expect(t.read).toBe((c.n * c.d + tr * 2 * c.n * c.d) * 2);
			expect(t.written).toBe(c.n * c.d * 2);
		}
	});

	it('the step counters add up to the closed form', () => {
		for (const c of every) {
			const run = simulate(c);
			const last = run.frames[run.frames.length - 1];
			const t = flashTraffic(c);
			expect(last.read).toBe(t.read);
			expect(last.written).toBe(t.written);
			expect(run.frames[0].read).toBe(0);
		}
	});

	it('causal reads less K/V than non-causal', () => {
		const a = flashTraffic(cfg({ n: 16, br: 2, bc: 2 }));
		const b = flashTraffic(cfg({ n: 16, br: 2, bc: 2, causal: true }));
		expect(b.tiles).toBe(36);
		expect(b.skipped).toBe(28);
		expect(b.read).toBeLessThan(a.read);
	});

	it('at a realistic size flash moves far fewer bytes; at tiny blocks it can move more', () => {
		const f = flashTraffic(REAL_CONFIG);
		const nv = naiveTraffic(REAL_CONFIG);
		// Q 512 KiB + 32 passes over K and V (32 MiB) + O 512 KiB, against 64 MiB of S and P each way.
		expect(f.read + f.written).toBe(33 * 2 ** 20);
		expect(nv.read + nv.written).toBe(130 * 2 ** 20);
		const tiny = cfg({ n: 16, d: 8, br: 2, bc: 2 });
		expect(flashTraffic(tiny).read + flashTraffic(tiny).written).toBeGreaterThan(naiveTraffic(tiny).read + naiveTraffic(tiny).written);
	});

	it('on-chip working set', () => {
		// (4x4 + 2x4x4) x 2 + 4x4x4 + 4x4x4 + 2x4x4 = 96 + 64 + 64 + 32.
		expect(onChip(cfg())).toEqual({ tiles: 96, scores: 64, acc: 64, stats: 32, total: 256 });
	});
});

suite('text', () => {
	it('formats', () => {
		expect(fmtBytes(1408)).toBe('1.38 KiB');
		expect(fmtBytes(640)).toBe('640 bytes');
		expect(fmtBytes(33 * 2 ** 20)).toBe('33 MiB');
		expect(num(-Infinity)).toBe('-inf');
		expect(num(-0.001)).toBe('0.00');
		expect(num(0.6789)).toBe('0.68');
	});

	it('describes every step of every setting', () => {
		for (const c of every) {
			const run = simulate(c);
			run.frames.forEach((_, s) => expect(describe(c, run, s).length).toBeGreaterThan(20));
		}
		const c = cfg();
		const run = simulate(c);
		expect(describe(c, run, 0)).toContain('Guess');
		expect(describe(c, run, run.frames.length - 1)).toContain('Done');
		expect(describe(c, run, 1)).toContain('First tile');
	});
});
