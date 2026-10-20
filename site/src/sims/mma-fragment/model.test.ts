import { describe as suite, expect, it } from 'vitest';
import {
	describe,
	elementOf,
	elementsOf,
	lanesForColOfB,
	lanesForRowOfA,
	LAST_STEP,
	ownerOf,
	pickText,
	registerOf,
	SHAPE,
	viewAt,
	WARP,
	type MatrixId,
} from './model';

const MATS: MatrixId[] = ['a', 'b', 'c'];

suite('PTX ISA m16n8k16 fragment layout', () => {
	it('every element is held by exactly one lane and slot', () => {
		for (const m of MATS) {
			const seen = new Set<string>();
			for (let lane = 0; lane < WARP; lane++) {
				for (let i = 0; i < SHAPE[m].perLane; i++) {
					const e = elementOf(m, lane, i);
					expect(e.row).toBeGreaterThanOrEqual(0);
					expect(e.row).toBeLessThan(SHAPE[m].rows);
					expect(e.col).toBeLessThan(SHAPE[m].cols);
					seen.add(`${e.row},${e.col}`);
				}
			}
			expect(seen.size).toBe(SHAPE[m].rows * SHAPE[m].cols);
		}
	});

	it('ownerOf inverts elementOf', () => {
		for (const m of MATS) {
			for (let lane = 0; lane < WARP; lane++) {
				for (let i = 0; i < SHAPE[m].perLane; i++) {
					const e = elementOf(m, lane, i);
					expect(ownerOf(m, e.row, e.col)).toEqual({ lane, i });
				}
			}
		}
	});

	it('lane 0 matches the figures', () => {
		// Figure "fragment layout for matrix A": T0 holds a0,a1 at row 0 cols 0,1; a2,a3 at row 8; a4..a7 at cols 8,9.
		expect(elementsOf('a', 0)).toEqual([
			{ row: 0, col: 0 },
			{ row: 0, col: 1 },
			{ row: 8, col: 0 },
			{ row: 8, col: 1 },
			{ row: 0, col: 8 },
			{ row: 0, col: 9 },
			{ row: 8, col: 8 },
			{ row: 8, col: 9 },
		]);
		expect(elementsOf('b', 0)).toEqual([
			{ row: 0, col: 0 },
			{ row: 1, col: 0 },
			{ row: 8, col: 0 },
			{ row: 9, col: 0 },
		]);
		expect(elementsOf('c', 0)).toEqual([
			{ row: 0, col: 0 },
			{ row: 0, col: 1 },
			{ row: 8, col: 0 },
			{ row: 8, col: 1 },
		]);
	});

	it('lane 5 and lane 31 by the formulas', () => {
		// Lane 5: groupID 1, threadID_in_group 1.
		expect(elementsOf('a', 5).map((e) => [e.row, e.col])).toEqual([
			[1, 2], [1, 3], [9, 2], [9, 3], [1, 10], [1, 11], [9, 10], [9, 11],
		]);
		expect(elementsOf('b', 5).map((e) => [e.row, e.col])).toEqual([[2, 1], [3, 1], [10, 1], [11, 1]]);
		// Lane 31: groupID 7, threadID_in_group 3.
		expect(elementsOf('c', 31).map((e) => [e.row, e.col])).toEqual([[7, 6], [7, 7], [15, 6], [15, 7]]);
		expect(elementOf('a', 31, 7)).toEqual({ row: 15, col: 15 });
		expect(elementOf('b', 31, 3)).toEqual({ row: 15, col: 7 });
	});

	it('registers pack two f16 per register for A and B', () => {
		expect(registerOf('a', 0)).toEqual({ reg: 0, half: 'low' });
		expect(registerOf('a', 7)).toEqual({ reg: 3, half: 'high' });
		expect(registerOf('b', 2)).toEqual({ reg: 1, half: 'low' });
		expect(registerOf('c', 3)).toEqual({ reg: 3, half: null });
	});

	it('a row of A lives in 4 lanes, a column of B in 4 lanes', () => {
		expect(lanesForRowOfA(0)).toEqual([0, 1, 2, 3]);
		expect(lanesForRowOfA(9)).toEqual([4, 5, 6, 7]);
		expect(lanesForColOfB(1)).toEqual([4, 5, 6, 7]);
		expect(lanesForColOfB(7)).toEqual([28, 29, 30, 31]);
	});
});

suite('stepper', () => {
	it('steps 4 to 7 compute the lane\'s four outputs', () => {
		const c = { lane: 5, pick: null };
		expect(viewAt(c, 0).target).toBeNull();
		expect(viewAt(c, 4).target).toEqual({ row: 1, col: 2 });
		expect(viewAt(c, 7).target).toEqual({ row: 9, col: 3 });
		expect(viewAt(c, 4).feeders).toEqual({ a: [4, 5, 6, 7], b: [8, 9, 10, 11] });
		for (let s = 0; s <= LAST_STEP; s++) expect(describe(c, s)).toMatch(new RegExp(`^Step ${s}`));
	});

	it('pick text', () => {
		expect(pickText({ m: 'a', row: 9, col: 11 })).toBe('A(9, 11) is held by lane 5 as a7, in a register 3, high half.');
		expect(pickText({ m: 'c', row: 15, col: 7 })).toBe('C/D(15, 7) is held by lane 31 as c3, in d/c register 3.');
	});
});
