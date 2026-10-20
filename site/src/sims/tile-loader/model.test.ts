import { describe as suite, expect, it } from 'vitest';
import { blockTotals, CODE, describe, gridTotals, lastStep, MODES, SIZES, stateAt, steps, TILES, type Config } from './model';

const all = MODES.flatMap((mode) => SIZES.flatMap((n) => TILES.map((tile) => ({ mode, n, tile }) as Config)));

suite('tile loader', () => {
	it('n = 8, T = 4: 256 global loads per block without shared memory, 64 with', () => {
		const g: Config = { mode: 'global', n: 8, tile: 4 };
		const s: Config = { mode: 'shared', n: 8, tile: 4 };
		expect(stateAt(g, lastStep(g)).globalReads).toBe(16 * 2 * 8);
		expect(stateAt(s, lastStep(s)).globalReads).toBe(2 * 2 * 16);
		expect(stateAt(s, lastStep(s)).sharedReads).toBe(256);
	});

	it.each(all)('final counts match the closed forms: %o', (c) => {
		const end = stateAt(c, lastStep(c));
		const b = blockTotals(c);
		expect(end.globalReads).toBe(c.mode === 'global' ? b.globalWithout : b.globalWith);
		expect(end.sharedReads).toBe(c.mode === 'global' ? 0 : b.shared);
		// Every thread adds n terms either way.
		expect(end.terms).toBe(c.n);
		expect(end.stored).toBe(true);
		// The per-element counts add up to the total.
		expect(end.aCount.reduce((x, y) => x + y) + end.bCount.reduce((x, y) => x + y)).toBe(end.globalReads);
	});

	it.each(all)('each strip element is loaded T times without shared memory, once with: %o', (c) => {
		const end = stateAt(c, lastStep(c));
		const used = end.aCount.filter((x) => x > 0);
		expect(used.length).toBe(c.tile * c.n);
		expect(new Set(used)).toEqual(new Set([c.mode === 'global' ? c.tile : 1]));
	});

	it('the saving is a factor of T for the grid too', () => {
		for (const n of SIZES)
			for (const tile of TILES) {
				const g = gridTotals({ mode: 'shared', n, tile });
				expect(g.globalWithout).toBe(2 * n ** 3);
				expect(g.globalWith * tile).toBe(g.globalWithout);
			}
	});

	it('every shared slot is read T times per phase', () => {
		const c: Config = { mode: 'shared', n: 8, tile: 4 };
		const st = steps(c);
		const sync2 = st.findIndex((x) => x.kind === 'sync2');
		const s = stateAt(c, sync2);
		expect(s.asReads.every((x) => x === 4)).toBe(true);
		expect(s.bsReads.every((x) => x === 4)).toBe(true);
		// The next load resets the per-phase counts.
		expect(stateAt(c, sync2 + 1).asReads.every((x) => x === 0)).toBe(true);
	});

	it('step counts: phases x (T + 3) + 2 with shared memory, n + 2 without', () => {
		expect(lastStep({ mode: 'shared', n: 8, tile: 4 })).toBe(2 * 7 + 1);
		expect(lastStep({ mode: 'shared', n: 16, tile: 2 })).toBe(8 * 5 + 1);
		expect(lastStep({ mode: 'global', n: 16, tile: 2 })).toBe(17);
	});

	it.each(all)('every step has a status and valid code lines: %o', (c) => {
		const code = CODE[c.mode].split('\n');
		for (let s = 0; s <= lastStep(c); s++) {
			expect(describe(c, s).length).toBeGreaterThan(40);
			for (const line of stateAt(c, s).cur.lines) expect(code[line - 1].trim().length).toBeGreaterThan(1);
		}
		expect(describe(c, lastStep(c))).toContain('Done');
	});

	it('code lines point at the statements they describe', () => {
		const shared = CODE.shared.split('\n');
		const kinds = Object.fromEntries(steps({ mode: 'shared', n: 8, tile: 4 }).map((x) => [x.kind, x.lines]));
		expect(shared[kinds.load[0] - 1]).toContain('As[ty][tx] =');
		expect(shared[kinds.sync1[0] - 1]).toContain('__syncthreads');
		expect(shared[kinds.compute[1] - 1]).toContain('As[ty][k] * Bs[k][tx]');
		expect(shared[kinds.sync2[0] - 1]).toContain('__syncthreads');
		expect(shared[kinds.store[0] - 1]).toContain('C[row * n + col]');
		const global = CODE.global.split('\n');
		expect(global[5]).toContain('A[row * n + k] * B[k * n + col]');
	});

	it('status text for the default case', () => {
		const c: Config = { mode: 'shared', n: 8, tile: 4 };
		expect(describe(c, 1)).toBe(
			'Phase 0: every thread loads one element of A into As and one of B into Bs, 32 global loads for the whole block. Each element of the two tiles is loaded once. Global loads so far: 32.',
		);
		expect(describe(c, lastStep(c))).toContain('64 global loads instead of 256, 4 times fewer');
	});
});
