import { describe as suite, expect, it } from 'vitest';
import { describe, gridDim, lastStep, stateAt, threads, uncovered, type Config } from './model';

const base: Config = { n: 20, blockDim: 8, gridMode: 'ceil', guard: true };

suite('grid indexing model', () => {
	it('ceil-div launches enough blocks, floor-div does not', () => {
		expect(gridDim(base)).toBe(3);
		expect(gridDim({ ...base, gridMode: 'floor' })).toBe(2);
		expect(uncovered({ ...base, gridMode: 'floor' })).toEqual([16, 17, 18, 19]);
		expect(uncovered(base)).toEqual([]);
	});

	it('maps every in-bounds index to exactly one thread', () => {
		const ts = threads(base).filter((t) => t.inBounds);
		expect(ts.map((t) => t.i)).toEqual([...Array(20).keys()]);
		expect(threads(base).filter((t) => !t.inBounds)).toHaveLength(4);
	});

	it('writes every element by the last step', () => {
		const end = stateAt(base, lastStep(base));
		expect(end.writtenBy).toEqual([...Array(20)].map((_, i) => Math.floor(i / 8)));
		expect(end.outOfBounds).toEqual([]);
	});

	it('steps through index, guard, write per block', () => {
		expect(lastStep(base)).toBe(9);
		expect(stateAt(base, 1)).toMatchObject({ block: 0, phase: 'index', lines: [2] });
		expect(stateAt(base, 2)).toMatchObject({ block: 0, phase: 'guard', lines: [3] });
		expect(stateAt(base, 3).writtenBy.filter((b) => b === 0)).toHaveLength(8);
		expect(stateAt(base, 4).writtenBy.filter((b) => b !== null)).toHaveLength(8);
	});

	it('without the guard, the last block writes past the end', () => {
		const c = { ...base, guard: false };
		expect(lastStep(c)).toBe(6);
		expect(stateAt(c, lastStep(c)).outOfBounds).toEqual([20, 21, 22, 23]);
		expect(describe(c, 6)).toContain('undefined behavior');
	});

	it('handles n smaller than one block and a grid of zero blocks', () => {
		const small = { ...base, n: 3 };
		expect(gridDim(small)).toBe(1);
		expect(stateAt(small, lastStep(small)).writtenBy).toEqual([0, 0, 0]);
		const empty = { ...small, gridMode: 'floor' as const };
		expect(lastStep(empty)).toBe(0);
		expect(describe(empty, 0)).toContain('no blocks launch');
	});
});
