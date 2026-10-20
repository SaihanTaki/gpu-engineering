import { describe as suite, expect, it } from 'vitest';
import {
	applyStage,
	CODE,
	ACTIVE_LINES,
	comparatorCount,
	describe,
	INPUT_IDS,
	inputKeys,
	isSorted,
	launchCounts,
	lastStep,
	stageCount,
	stages,
	stateAt,
	type Config,
} from './model';

const configs: Config[] = [8, 16].flatMap((n) => INPUT_IDS.map((input) => ({ n, input, seed: 17 }) as Config));

suite('bitonic network', () => {
	it('8 keys: 6 stages, 24 comparators; 16 keys: 10 stages, 80 comparators', () => {
		expect([stageCount(8), comparatorCount(8)]).toEqual([6, 24]);
		expect([stageCount(16), comparatorCount(16)]).toEqual([10, 80]);
		expect(stages(8).map((s) => [s.k, s.j])).toEqual([
			[2, 1],
			[4, 2],
			[4, 1],
			[8, 4],
			[8, 2],
			[8, 1],
		]);
	});

	it.each([8, 16])('every stage touches each of the %i wires exactly once', (n) => {
		for (const st of stages(n)) {
			const wires = st.comparators.flatMap((c) => [c.i, c.p]).sort((a, b) => a - b);
			expect(wires).toEqual(Array.from({ length: n }, (_, i) => i));
			expect(st.comparators.every((c) => c.p - c.i === st.j)).toBe(true);
		}
	});

	it('the last phase sorts every pair ascending', () => {
		const all = stages(16);
		for (const st of all.slice(-4)) expect(st.comparators.every((c) => c.up)).toBe(true);
		// Phase k = 2: pairs (0,1) up, (2,3) down, ...
		expect(all[0].comparators.map((c) => c.up)).toEqual([true, false, true, false, true, false, true, false]);
	});

	it.each([8, 16])('sorts all 2^%i inputs of zeros and ones (the 0-1 principle)', (n) => {
		const net = stages(n);
		let unsorted = 0;
		for (let bits = 0; bits < 1 << n; bits++) {
			let keys = Array.from({ length: n }, (_, i) => (bits >> i) & 1);
			for (const st of net) keys = applyStage(keys, st).keys;
			if (!isSorted(keys)) unsorted++;
		}
		expect(unsorted).toBe(0);
	});

	it.each(configs)('ends sorted, with the same comparator count for any input: %o', (c) => {
		const end = stateAt(c, lastStep(c));
		expect(end.keys).toEqual(Array.from({ length: c.n }, (_, i) => i + 1));
		expect(end.comparatorsSoFar).toBe(comparatorCount(c.n));
		expect(inputKeys(c).slice().sort((a, b) => a - b)).toEqual(end.keys);
	});

	it('sorted input still swaps: the early phases build descending runs', () => {
		const c: Config = { n: 8, input: 'sorted', seed: 0 };
		expect(stateAt(c, 1).swapped.filter(Boolean).length).toBe(2);
		expect(stateAt(c, lastStep(c)).swapsSoFar).toBeGreaterThan(0);
	});

	it('random input is a fixed permutation for a seed', () => {
		const a = inputKeys({ n: 16, input: 'random', seed: 17 });
		expect(a).toEqual(inputKeys({ n: 16, input: 'random', seed: 17 }));
		expect(isSorted(a)).toBe(false);
	});

	it.each(configs)('every step has a status line: %o', (c) => {
		for (let s = 0; s <= lastStep(c); s++) expect(describe(c, s).length).toBeGreaterThan(60);
		expect(describe(c, lastStep(c))).toContain('sorted');
	});

	it('the highlighted lines are the launch and the compare-swap', () => {
		const lines = CODE.split('\n');
		expect(lines[ACTIVE_LINES[0] - 1]).toContain('bitonic_step<<<');
		expect(lines[ACTIVE_LINES[1] - 1]).toContain('compare_swap');
	});
});

suite('launch counts', () => {
	it('2^20 keys, tile 1024: 210 launches per stage, or 55 global + 11 shared', () => {
		expect(launchCounts(1 << 20, 1024)).toEqual({ perStage: 210, hybridGlobal: 55, hybridShared: 11 });
	});
	it('one tile needs a single shared memory launch', () => {
		expect(launchCounts(1024, 1024)).toEqual({ perStage: 55, hybridGlobal: 0, hybridShared: 1 });
	});
});
