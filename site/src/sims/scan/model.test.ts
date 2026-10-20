import { describe as suite, expect, it } from 'vitest';
import { ALGO_IDS, countsAt, depth, describe, lastStep, linesAt, MODES, phases, reference, SIZES, valuesAt, work, type Config } from './model';

const cfg = (over: Partial<Config> = {}): Config => ({ n: 8, mode: 'exclusive', input: 'mixed', ...over });
const all = SIZES.flatMap((n) => MODES.flatMap((mode) => (['mixed', 'ones'] as const).map((input) => cfg({ n, mode, input }))));

suite('results', () => {
	it("Blelloch's example: 3 1 7 0 4 1 6 3", () => {
		expect(reference(cfg())).toEqual([0, 3, 4, 11, 11, 15, 16, 22]);
		expect(reference(cfg({ mode: 'inclusive' }))).toEqual([3, 4, 11, 11, 15, 16, 22, 25]);
	});

	it('both algorithms end at the reference scan for every setting', () => {
		for (const c of all) for (const a of ALGO_IDS) expect(valuesAt(c, a, lastStep(c))).toEqual(reference(c));
	});

	it('Hillis-Steele on all ones after step k holds min(i + 1, 2^k)', () => {
		const ps = phases(cfg({ input: 'ones', mode: 'inclusive' }), 'hs');
		expect(ps[0].values).toEqual([1, 2, 2, 2, 2, 2, 2, 2]);
		expect(ps[1].values).toEqual([1, 2, 3, 4, 4, 4, 4, 4]);
	});

	it('Blelloch up-sweep leaves the total in the last element, then clears it', () => {
		const ps = phases(cfg(), 'blelloch');
		expect(ps[2].values).toEqual([3, 4, 7, 11, 4, 5, 6, 25]);
		expect(ps[3].kind).toBe('clear');
		expect(ps[3].values[7]).toBe(0);
	});
});

suite('work and depth', () => {
	it('n = 8: Hillis-Steele 7 + 6 + 4 = 17 additions in 3 levels, Blelloch 14 in 6', () => {
		const end = lastStep(cfg());
		expect(countsAt(cfg(), 'hs', end)).toMatchObject({ adds: 17, addLevels: 3, done: true });
		expect(countsAt(cfg(), 'blelloch', end)).toMatchObject({ adds: 14, addLevels: 6, done: true });
	});

	it('the phase sums match the closed forms, and the result type adds no work', () => {
		for (const c of all) {
			const end = lastStep(c);
			expect(countsAt(c, 'hs', end).adds).toBe(work.hs(c.n));
			expect(countsAt(c, 'blelloch', end).adds).toBe(work.blelloch(c.n));
			expect(countsAt(c, 'hs', end).addLevels).toBe(depth.hs(c.n));
			expect(countsAt(c, 'blelloch', end).addLevels).toBe(depth.blelloch(c.n));
		}
		// 16 * 4 - 15 = 49; 2 * 15 = 30.
		expect(work.hs(16)).toBe(49);
		expect(work.blelloch(16)).toBe(30);
		expect(work.hs(1024)).toBe(9217);
	});

	it('step counts: the shorter algorithm waits at its result', () => {
		expect(phases(cfg({ mode: 'inclusive' }), 'hs')).toHaveLength(3);
		expect(phases(cfg(), 'hs')).toHaveLength(4);
		expect(phases(cfg(), 'blelloch')).toHaveLength(7);
		expect(phases(cfg({ mode: 'inclusive' }), 'blelloch')).toHaveLength(8);
		expect(lastStep(cfg({ n: 16, mode: 'inclusive' }))).toBe(10);
		expect(countsAt(cfg(), 'hs', 6)).toMatchObject({ phases: 4, done: true });
	});

	it('down-sweep pairs do one copy and one addition each', () => {
		const down = phases(cfg(), 'blelloch').filter((p) => p.kind === 'down');
		expect(down.map((p) => p.adds)).toEqual([1, 2, 4]);
		expect(down[0].ops).toEqual([
			{ kind: 'copy', from: 7, to: 3 },
			{ kind: 'add', from: 3, to: 7 },
		]);
	});
});

suite('status and code', () => {
	it('every step has a status line', () => {
		for (const c of all) for (let s = 0; s <= lastStep(c); s++) expect(describe(c, s).length).toBeGreaterThan(40);
		expect(describe(cfg(), 0)).toContain('7 additions');
		expect(describe(cfg(), 5)).toContain('Hillis-Steele is done: 17 additions in 3 addition levels.');
		expect(describe(cfg(), 7)).toContain('Result: 0, 3, 4, 11, 11, 15, 16, 22.');
	});

	it('maps phases to code lines', () => {
		expect([0, 1, 4, 5].map((s) => linesAt(cfg(), 'hs', s))).toEqual([[1], [3, 5], [8, 9], []]);
		expect([1, 4, 5, 7].map((s) => linesAt(cfg(), 'blelloch', s))).toEqual([[6], [9], [13], [13]]);
	});
});
