import { describe as suite, expect, it } from 'vitest';
import { describe, simulate, stream, totals, type Config } from './model';

const base: Config = { pattern: 'sequential', param: 1, accesses: 128, l1Lines: 4, l2Lines: 16 };
const run = (c: Partial<Config>) => {
	const cfg = { ...base, ...c };
	const sim = simulate(cfg);
	return totals(sim, sim.length);
};

// 8 floats per 32-byte line throughout; the comments show the arithmetic.
suite('hand-computed totals', () => {
	it('sequential: one miss per line, then 7 hits', () => {
		// 128 accesses / 8 per line = 16 lines, each read once from DRAM.
		expect(run({})).toEqual({ accesses: 128, l1: 112, l2: 0, dram: 16, dramBytes: 512 });
	});

	it('stride 8: every access is a new line', () => {
		expect(run({ pattern: 'strided', param: 8 })).toMatchObject({ l1: 0, l2: 0, dram: 128 });
	});

	it('stride 2: a new line every 4 accesses', () => {
		expect(run({ pattern: 'strided', param: 2 })).toMatchObject({ l1: 96, l2: 0, dram: 32 });
	});

	it('window that fits L1: only the first pass misses', () => {
		// 32 elements = 4 lines = all of L1. 4 passes; 4 misses in the first.
		expect(run({ pattern: 'window', param: 32 })).toMatchObject({ l1: 124, l2: 0, dram: 4 });
	});

	it('window that fits L2 but not L1: later passes hit L2 once per line', () => {
		// 64 elements = 8 lines. LRU over 4 L1 lines evicts each line before it comes back.
		// 3 passes: 8 DRAM misses, then 8 L2 hits per pass, 7 L1 hits per line throughout.
		expect(run({ pattern: 'window', param: 64, accesses: 192 })).toMatchObject({ l1: 168, l2: 16, dram: 8 });
	});

	it('window one line bigger than L1 still thrashes L1 under LRU', () => {
		// 40 elements = 5 lines, 2 passes.
		expect(run({ pattern: 'window', param: 40, accesses: 80 })).toMatchObject({ l1: 70, l2: 5, dram: 5 });
	});

	it('window bigger than L2: LRU evicts every line before its reuse', () => {
		// 160 elements = 20 lines > 16. Both passes miss every line.
		expect(run({ pattern: 'window', param: 160, accesses: 320 })).toMatchObject({ l1: 280, l2: 0, dram: 40 });
	});

	it('random over one line: a single miss', () => {
		expect(run({ pattern: 'random', param: 8 })).toMatchObject({ l1: 127, l2: 0, dram: 1 });
	});
});

suite('stream and cache state', () => {
	it('random streams are deterministic and in range', () => {
		const c = { ...base, pattern: 'random' as const, param: 1024 };
		const s = stream(c);
		expect(s).toEqual(stream(c));
		expect(Math.min(...s)).toBeGreaterThanOrEqual(0);
		expect(Math.max(...s)).toBeLessThan(1024);
		const t = run(c);
		expect(t.l1 + t.l2 + t.dram).toBe(128);
	});

	it('keeps lines most recent first and evicts the least recent', () => {
		const sim = simulate({ ...base, accesses: 40 });
		expect(sim[15].l1).toEqual([1, 0]);
		// Access 33 reads element 32, line 4: L1 is full with lines 3, 2, 1, 0.
		expect(sim[32]).toMatchObject({ line: 4, outcome: 'dram', l1: [4, 3, 2, 1], l1Evicted: 0, l2Evicted: null });
	});

	it('an L2 hit refreshes L2 order and fills L1', () => {
		const sim = simulate({ ...base, pattern: 'window', param: 40, accesses: 41 });
		// Access 41 reads line 0 again: L1 has 4, 3, 2, 1; L2 still has all 5.
		expect(sim[40]).toMatchObject({ line: 0, outcome: 'l2', l1: [0, 4, 3, 2], l1Evicted: 1, l2: [0, 4, 3, 2, 1] });
	});
});

suite('status text', () => {
	const c = { ...base, accesses: 40 };
	const sim = simulate(c);
	it('step 0 invites a guess', () => {
		expect(describe(c, sim, 0)).toMatch(/Guess the L1 hit rate/);
	});
	it('names hits, misses and evictions', () => {
		expect(describe(c, sim, 2)).toBe('Access 2 of 40: a[1], byte 4, line 0. L1 hit. So far: L1 hits 1, L2 hits 0, DRAM reads 1.');
		expect(describe(c, sim, 33)).toMatch(/Miss in L1 and L2: line 4 read from DRAM \(32 bytes\) into L2 and L1; L1 evicts line 0\./);
	});
});
