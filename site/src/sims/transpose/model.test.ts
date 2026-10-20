import { describe as suite, expect, it } from 'vitest';
import { gpuById } from '@sim-core/gpu-specs';
import {
	accesses,
	accessIndexAt,
	clockGhz,
	describe,
	estimate,
	globalCost,
	KERNEL_IDS,
	LAST_STEP,
	lineAt,
	sharedCost,
	totalsAt,
	type Config,
} from './model';

const cfg = (over: Partial<Config> = {}): Config => ({ gpu: 'a100', n: 1024, tile: 'first', row: 0, ...over });
const kinds = (c: Config, k: (typeof KERNEL_IDS)[number]) => accesses(c, k).map((a) => a.kind);

suite('sectors per warp request', () => {
	it('a row of 32 floats at an aligned address is 4 sectors, fully used', () => {
		const g = globalCost(accesses(cfg(), 'naive')[0]);
		expect(g.sectors.map((s) => s.sector)).toEqual([0, 1, 2, 3]);
		expect(g.sectors.every((s) => s.words.every(Boolean))).toBe(true);
		expect(g.usedBytes).toBe(128);
	});

	it('the naive store writes a column: 32 sectors, 4 of 32 bytes used in each', () => {
		const g = globalCost(accesses(cfg(), 'naive')[1]);
		expect(g.sectors).toHaveLength(32);
		expect(g.sectors.every((s) => s.words.filter(Boolean).length === 1)).toBe(true);
	});

	it('the tiled kernels store a row of out: 4 sectors', () => {
		for (const k of ['tiled', 'padded'] as const) expect(globalCost(accesses(cfg(), k)[3]).sectors).toHaveLength(4);
	});

	it('n = 1001 shifts row 3 off a sector boundary: 5 sectors', () => {
		// Row 3 starts at element 3003; 3003 / 8 = 375.4 and 3034 / 8 = 379.3, so sectors 375..379.
		const g = globalCost(accesses(cfg({ n: 1001, row: 3 }), 'naive')[0]);
		expect(g.sectors.map((s) => s.sector)).toEqual([375, 376, 377, 378, 379]);
		expect(globalCost(accesses(cfg({ n: 1001, row: 0 }), 'naive')[0]).sectors).toHaveLength(4);
	});
});

suite('bank conflicts', () => {
	it('writing a row of the tile hits 32 different banks', () => {
		for (const k of ['tiled', 'padded'] as const) expect(sharedCost(accesses(cfg(), k)[1]).degree).toBe(1);
	});

	it('reading a column of a 32-wide tile puts every lane in one bank', () => {
		// Word lane * 32 + 5 is in bank 5 for every lane.
		const s = sharedCost(accesses(cfg({ row: 5 }), 'tiled')[2]);
		expect(s.degree).toBe(32);
		expect(s.perBank[5]).toBe(32);
	});

	it('padding to 33 words spreads the column over all 32 banks', () => {
		// Word lane * 33 + r is in bank (lane + r) % 32.
		const s = sharedCost(accesses(cfg({ row: 5 }), 'padded')[2]);
		expect(s.degree).toBe(1);
		expect(s.perBank.every((n) => n === 1)).toBe(true);
	});

	it('lanes reading the same word are one pass (broadcast)', () => {
		expect(sharedCost({ kind: 'shared-load', index: new Array(32).fill(7) }).degree).toBe(1);
	});
});

suite('edge tiles', () => {
	it('n = 1000, last tile: only 8 columns are inside the matrix', () => {
		// Tile 31 covers columns 992..1023; 992..999 exist.
		const load = accesses(cfg({ n: 1000, tile: 'last' }), 'tiled')[0];
		expect(load.index.filter((i) => i !== null)).toHaveLength(8);
		// 992 * 1000 + 992 = 992992, a multiple of 8: one sector.
		expect(globalCost(load).sectors).toHaveLength(1);
	});

	it('n = 1000, last tile, row 8 is past the end: nothing moves', () => {
		const c = cfg({ n: 1000, tile: 'last', row: 8 });
		for (const k of KERNEL_IDS) expect(totalsAt(c, k, LAST_STEP)).toEqual({ loadSectors: 0, storeSectors: 0, sharedPasses: 0, usefulBytes: 0, sectorBytes: 0 });
		expect(estimate(gpuById('a100'), totalsAt(c, 'naive', LAST_STEP)).gbPerS).toBeNull();
	});
});

suite('steps', () => {
	it('naive has two accesses, the tiled kernels four', () => {
		expect(kinds(cfg(), 'naive')).toEqual(['global-load', 'global-store']);
		expect(kinds(cfg(), 'tiled')).toEqual(['global-load', 'shared-store', 'shared-load', 'global-store']);
	});

	it('maps steps to accesses and code lines', () => {
		expect([0, 1, 2, 3, 4, 5, 6].map((s) => accessIndexAt('naive', s))).toEqual([null, 0, 1, null, null, null, null]);
		expect([0, 1, 2, 3, 4, 5, 6].map((s) => accessIndexAt('tiled', s))).toEqual([null, 0, 1, null, 2, 3, null]);
		expect([1, 2, 3, 4, 5].map((s) => lineAt('padded', s))).toEqual([6, 6, 7, 12, 12]);
	});

	it('totals at the end', () => {
		expect(totalsAt(cfg(), 'naive', LAST_STEP)).toEqual({ loadSectors: 4, storeSectors: 32, sharedPasses: 0, usefulBytes: 256, sectorBytes: 1152 });
		expect(totalsAt(cfg(), 'tiled', LAST_STEP)).toEqual({ loadSectors: 4, storeSectors: 4, sharedPasses: 33, usefulBytes: 256, sectorBytes: 256 });
		expect(totalsAt(cfg(), 'padded', LAST_STEP)).toEqual({ loadSectors: 4, storeSectors: 4, sharedPasses: 2, usefulBytes: 256, sectorBytes: 256 });
		expect(totalsAt(cfg(), 'tiled', 2).sharedPasses).toBe(1);
	});

	it('every step has a status line', () => {
		for (let s = 0; s <= LAST_STEP; s++) expect(describe(cfg(), s).length).toBeGreaterThan(20);
		expect(describe(cfg(), 4)).toContain('32-way bank conflict');
	});
});

suite('bandwidth estimate', () => {
	it('derives clocks from peak FP32', () => {
		// 19.5e12 / (108 * 64 * 2) = 1.4106 GHz; 8.1e12 / (40 * 64 * 2) = 1.5820; 67e12 / (132 * 128 * 2) = 1.9827.
		expect(clockGhz(gpuById('a100'))).toBeCloseTo(1.4106, 4);
		expect(clockGhz(gpuById('t4'))).toBeCloseTo(1.582, 3);
		expect(clockGhz(gpuById('h100'))).toBeCloseTo(1.9827, 4);
	});

	it('A100: naive 22% of peak, tiled limited by shared memory, padded at peak', () => {
		const g = gpuById('a100');
		const e = (k: (typeof KERNEL_IDS)[number]) => estimate(g, totalsAt(cfg(), k, LAST_STEP));
		// 256 / 1152 * 2039 = 453.1.
		expect(e('naive').gbPerS).toBeCloseTo(453.1, 1);
		expect(e('naive').limit).toBe('global');
		// 33 passes / (108 SMs * 1.4106 GHz) = 0.21662 ns > 256 / 2039 = 0.12555 ns; 256 / 0.21662 = 1181.8.
		expect(e('tiled').limit).toBe('shared');
		expect(e('tiled').gbPerS).toBeCloseTo(1181.8, 0);
		expect(e('padded').gbPerS).toBeCloseTo(2039, 6);
		expect(e('padded').limit).toBe('global');
	});

	it('T4: the conflicts do not set the time in this model', () => {
		// 33 / (40 * 1.582) = 0.5215 ns < 256 / 300 = 0.8533 ns.
		const e = estimate(gpuById('t4'), totalsAt(cfg(), 'tiled', LAST_STEP));
		expect(e.limit).toBe('global');
		expect(e.gbPerS).toBeCloseTo(300, 6);
	});
});
