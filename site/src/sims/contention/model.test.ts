import { describe as suite, expect, it } from 'vitest';
import {
	BINS,
	BLOCK_OPTIONS,
	describe,
	DIST_IDS,
	globalHistAt,
	globalQueues,
	histogram,
	HOT_BIN,
	input,
	LAST_STEP,
	lineAt,
	MERGE_STEP,
	mergeQueues,
	N,
	privateCounts,
	ROUNDS,
	sharedQueues,
	stepSlots,
	THREADS,
	totalsAt,
	type Config,
	type Dist,
} from './model';

const cfg = (dist: Dist, blocks = 4): Config => ({ dist, blocks });
const all = DIST_IDS.flatMap((d) => BLOCK_OPTIONS.map((b) => cfg(d, b)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

suite('input', () => {
	it('every distribution has N elements in range', () => {
		for (const d of DIST_IDS) {
			expect(input(d)).toHaveLength(N);
			expect(input(d).every((b) => Number.isInteger(b) && b >= 0 && b < BINS)).toBe(true);
		}
	});

	it('the hot input puts every element in one bin', () => {
		expect(histogram('hot')[HOT_BIN]).toBe(N);
	});

	it('the skewed input puts about half in bin 0 and fewer in each later bin', () => {
		const h = histogram('skewed');
		expect(h[0]).toBeGreaterThan(N / 2 - 16);
		expect(h[0]).toBeLessThan(N / 2 + 16);
		expect(h[0]).toBeGreaterThan(h[1]);
		expect(h[1]).toBeGreaterThan(h[2]);
	});

	it('is the same on every call', () => {
		expect(input('uniform')).toEqual(input('uniform'));
	});
});

suite('queues', () => {
	it('a global round queues every thread on its bin', () => {
		for (const c of all) {
			for (let r = 1; r <= ROUNDS; r++) expect(sum(globalQueues(c, r).map((q) => q.length))).toBe(THREADS);
		}
	});

	it('one hot bin: 32 threads queue on one counter, 32 slots per round', () => {
		const c = cfg('hot');
		expect(globalQueues(c, 1)[HOT_BIN]).toHaveLength(32);
		expect(stepSlots(c, 'global', 1)).toBe(32);
		expect(totalsAt(c, 'global', LAST_STEP).slots).toBe(N);
	});

	it('privatizing splits the hot queue by the number of blocks', () => {
		for (const blocks of BLOCK_OPTIONS) {
			const c = cfg('hot', blocks);
			expect(stepSlots(c, 'shared', 1)).toBe(THREADS / blocks);
			// Every block merges one nonzero bin, so the merge queue is one per block.
			expect(stepSlots(c, 'shared', MERGE_STEP)).toBe(blocks);
			expect(totalsAt(c, 'shared', LAST_STEP).slots).toBe((ROUNDS * THREADS) / blocks + blocks);
		}
	});

	it('a block only queues its own threads', () => {
		const c = cfg('uniform', 4);
		sharedQueues(c, 2).forEach((row, blk) => row.flat().forEach((t) => expect(Math.floor(t / 8)).toBe(blk)));
	});

	it('with one block, the shared kernel needs one slot more than the global one on the hot input', () => {
		const c = cfg('hot', 1);
		expect(totalsAt(c, 'shared', LAST_STEP).slots).toBe(totalsAt(c, 'global', LAST_STEP).slots + 1);
	});

	it('the merge adds only nonzero bins', () => {
		for (const c of all) {
			const counts = privateCounts(c, ROUNDS);
			mergeQueues(c).forEach((q, b) => expect(q.length).toBe(counts.filter((row) => row[b] > 0).length));
		}
	});
});

suite('totals', () => {
	it('both kernels end with the same histogram and the right atomic counts', () => {
		for (const c of all) {
			const h = histogram(c.dist);
			expect(globalHistAt(c, 'global', LAST_STEP)).toEqual(h);
			expect(globalHistAt(c, 'shared', LAST_STEP)).toEqual(h);
			expect(sum(privateCounts(c, ROUNDS).map(sum))).toBe(N);
			const g = totalsAt(c, 'global', LAST_STEP);
			const s = totalsAt(c, 'shared', LAST_STEP);
			expect(g.globalAtomics).toBe(N);
			expect(g.perBin).toEqual(h);
			expect(s.sharedAtomics).toBe(N);
			expect(s.globalAtomics).toBe(sum(s.perBin));
			expect(s.globalAtomics).toBeLessThanOrEqual(c.blocks * BINS);
		}
	});

	it('the shared kernel writes global memory only at the merge', () => {
		const c = cfg('uniform');
		expect(globalHistAt(c, 'shared', ROUNDS)).toEqual(new Array(BINS).fill(0));
		expect(totalsAt(c, 'shared', ROUNDS).globalAtomics).toBe(0);
	});

	it('slots never shrink as steps advance', () => {
		for (const c of all) {
			for (const k of ['global', 'shared'] as const) {
				for (let s = 1; s <= LAST_STEP; s++) expect(totalsAt(c, k, s).slots).toBeGreaterThanOrEqual(totalsAt(c, k, s - 1).slots);
			}
		}
	});
});

suite('text and code', () => {
	it('every step has a status', () => {
		for (const c of all) for (let s = 0; s <= LAST_STEP; s++) expect(describe(c, s).length).toBeGreaterThan(20);
	});

	it('the last status gives both slot totals', () => {
		const text = describe(cfg('hot', 4), LAST_STEP);
		expect(text).toContain('Global: 128 global atomics, 128 slots');
		expect(text).toContain('Shared: 128 shared and 4 global atomics, 36 slots');
	});

	it('highlights the atomic line of each kernel', () => {
		expect(lineAt('global', 1)).toBe(3);
		expect(lineAt('shared', 1)).toBe(6);
		expect(lineAt('shared', MERGE_STEP)).toBe(9);
		expect(lineAt('global', MERGE_STEP)).toBeNull();
		expect(lineAt('global', 0)).toBeNull();
	});
});
