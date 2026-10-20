import { describe as suite, expect, it } from 'vitest';
import { CODE, describe, ITERATIONS, lastStep, linesAt, LIMITS, programLength, simulate, totalsAt, warpsToHide, type Config } from './model';

const cfg = (warps: number, latency: number, alu: number): Config => ({ warps, latency, alu });

const all: Config[] = [];
for (let w = LIMITS.warps.min; w <= LIMITS.warps.max; w++)
	for (let l = LIMITS.latency.min; l <= LIMITS.latency.max; l += 4)
		for (let a = LIMITS.alu.min; a <= LIMITS.alu.max; a++) all.push(cfg(w, l, a));

suite('scheduler', () => {
	it('one warp: every iteration costs latency + alu cycles', () => {
		const c = cfg(1, 20, 2);
		const t = totalsAt(c, lastStep(c));
		expect(t.cycles).toBe(ITERATIONS * (20 + 2));
		expect(t.issued).toBe(ITERATIONS * 3);
		expect(t.busy).toBeCloseTo(12 / 88, 10);
	});

	it.each(all)('issues each warp program exactly once, at most one instruction per cycle: %o', (c) => {
		const trace = simulate(c);
		const perWarp = Array(c.warps).fill(0);
		for (const cy of trace) if (cy.issued !== null) perWarp[cy.issued]++;
		expect(perWarp).toEqual(Array(c.warps).fill(programLength(c)));
		expect(trace.at(-1)!.issued).not.toBeNull();
	});

	it.each(all)('a warp never issues before its load returns: %o', (c) => {
		const lastLoad = Array(c.warps).fill(-Infinity);
		for (const cy of simulate(c)) {
			if (cy.issued === null) continue;
			if (cy.kind === 'alu') expect(cy.cycle - lastLoad[cy.issued]).toBeGreaterThanOrEqual(c.latency);
			else lastLoad[cy.issued] = cy.cycle;
		}
	});

	it.each(all)('never idles while a warp is ready: %o', (c) => {
		for (const cy of simulate(c)) if (cy.issued === null) expect(cy.warps).not.toContain('ready');
	});

	it.each(all.filter((c) => c.warps >= warpsToHide(c)))('with enough warps, no idle cycle from the first returned load until a warp finishes: %o', (c) => {
		const trace = simulate(c);
		const firstDone = trace.findIndex((cy) => cy.warps.includes('done'));
		expect(trace.slice(c.latency, firstDone).every((cy) => cy.issued !== null)).toBe(true);
	});

	it('more warps hide more of the latency', () => {
		const busy = [1, 2, 4, 8].map((w) => totalsAt(cfg(w, 20, 2), Infinity).busy);
		for (let k = 1; k < busy.length; k++) expect(busy[k]).toBeGreaterThan(busy[k - 1]);
		expect(warpsToHide(cfg(1, 20, 2))).toBe(8);
		expect(warpsToHide(cfg(1, 4, 6))).toBe(2);
	});
});

suite('text and code', () => {
	it('highlights the load line or the arithmetic line', () => {
		const lines = CODE.split('\n');
		expect(lines[4]).toContain('a[k * n + i]');
		expect(lines[5]).toContain('f(x)');
		const c = cfg(4, 20, 2);
		expect(linesAt(c, 0)).toEqual([]);
		expect(linesAt(c, 1)).toEqual([5]);
		expect(linesAt(c, lastStep(c))).toEqual([6]);
	});

	it('every step has a status line', () => {
		for (const c of [cfg(1, 20, 2), cfg(4, 20, 2), cfg(8, 40, 1)])
			for (let s = 0; s <= lastStep(c); s++) expect(describe(c, s).length).toBeGreaterThan(40);
		const c = cfg(4, 20, 2);
		expect(describe(c, 1)).toBe('Cycle 0: warp 0 issues the load of iteration 1. Its next instruction needs x, so it cannot issue again before cycle 20. Ready but not picked: warps 1, 2 and 3.');
		expect(describe(c, 5)).toContain('no warp is ready');
		expect(describe(c, 5)).toContain('Warps 0, 1, 2 and 3 wait on memory; the first result arrives at cycle 20.');
		expect(describe(c, lastStep(c))).toContain('It needs about 8 warps');
		expect(describe(cfg(8, 20, 2), lastStep(cfg(8, 20, 2)))).toContain('never idles once the loop is running');
	});
});
