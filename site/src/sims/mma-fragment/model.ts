/**
 * Which lane of a warp holds which matrix element for
 * mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, from the PTX ISA section
 * "Matrix Fragments for mma.m16n8k16 with floating point type" (9.7.16.5.8 in
 * PTX ISA 9.4). A is 16 x 16 (f16), B is 16 x 8 (f16), C and D are 16 x 8 (f32).
 */

export type MatrixId = 'a' | 'b' | 'c';

export const SHAPE: Record<MatrixId, { rows: number; cols: number; perLane: number; name: string }> = {
	a: { rows: 16, cols: 16, perLane: 8, name: 'A' },
	b: { rows: 16, cols: 8, perLane: 4, name: 'B' },
	c: { rows: 16, cols: 8, perLane: 4, name: 'C/D' },
};

export const M = 16;
export const N = 8;
export const K = 16;
export const WARP = 32;

export const groupId = (lane: number) => lane >> 2;
export const threadInGroup = (lane: number) => lane % 4;

export interface Element {
	row: number;
	col: number;
}

/** Row and column of fragment element i (a_i, b_i or c_i) held by lane, exactly as the PTX ISA states. */
export function elementOf(m: MatrixId, lane: number, i: number): Element {
	const g = groupId(lane);
	const t = threadInGroup(lane);
	if (m === 'a') {
		const row = (0 <= i && i < 2) || (4 <= i && i < 6) ? g : g + 8;
		const col = i < 4 ? t * 2 + (i & 1) : t * 2 + (i & 1) + 8;
		return { row, col };
	}
	if (m === 'b') {
		const row = i < 2 ? t * 2 + (i & 1) : t * 2 + (i & 1) + 8;
		return { row, col: g };
	}
	return { row: i < 2 ? g : g + 8, col: t * 2 + (i & 1) };
}

export interface Owner {
	lane: number;
	/** Index of the element in the lane's fragment: a0..a7, b0..b3, c0..c3. */
	i: number;
}

/** The inverse of elementOf: which lane and fragment slot hold (row, col). */
export function ownerOf(m: MatrixId, row: number, col: number): Owner {
	if (m === 'a') {
		const lane = (row % 8) * 4 + ((col % 8) >> 1);
		return { lane, i: (col & 1) + (row >= 8 ? 2 : 0) + (col >= 8 ? 4 : 0) };
	}
	if (m === 'b') {
		const lane = col * 4 + ((row % 8) >> 1);
		return { lane, i: (row & 1) + (row >= 8 ? 2 : 0) };
	}
	const lane = (row % 8) * 4 + (col >> 1);
	return { lane, i: (col & 1) + (row >= 8 ? 2 : 0) };
}

export const elementsOf = (m: MatrixId, lane: number) => Array.from({ length: SHAPE[m].perLane }, (_, i) => elementOf(m, lane, i));

/**
 * The register an element sits in. A and B pack two f16 values per 32-bit
 * register (.f16x2), low half first; C and D in f32 take one register each.
 */
export function registerOf(m: MatrixId, i: number) {
	if (m === 'c') return { reg: i, half: null as null | 'low' | 'high' };
	return { reg: i >> 1, half: (i & 1 ? 'high' : 'low') as 'low' | 'high' };
}

export const elemName = (m: MatrixId, i: number) => `${m}${i}`;

export function regText(m: MatrixId, i: number) {
	const r = registerOf(m, i);
	const name = `${m === 'c' ? 'd/c' : m} register ${r.reg}`;
	return r.half ? `${name}, ${r.half} half` : name;
}

/**
 * Stepper. 0: overview; 1: A; 2: B; 3: C; 4 to 7: d0..d3 of the selected lane,
 * each with the row of A and column of B the tensor core reads for it.
 */
export const LAST_STEP = 7;

export interface Config {
	lane: number;
	/** The element the reader picked, if any. */
	pick: { m: MatrixId; row: number; col: number } | null;
}

export interface View {
	/** Matrices drawn with ownership colors this step. */
	shown: MatrixId[];
	/** The output element being computed (steps 4 to 7). */
	target: Element | null;
	/** Lanes whose registers feed the target. */
	feeders: { a: number[]; b: number[] };
}

const uniq = (xs: number[]) => [...new Set(xs)].sort((p, q) => p - q);

export function lanesForRowOfA(row: number) {
	return uniq(Array.from({ length: K }, (_, k) => ownerOf('a', row, k).lane));
}
export function lanesForColOfB(col: number) {
	return uniq(Array.from({ length: K }, (_, k) => ownerOf('b', k, col).lane));
}

export function viewAt(c: Config, step: number): View {
	const shown: MatrixId[] = step === 0 ? ['a', 'b', 'c'] : step === 1 ? ['a'] : step === 2 ? ['b'] : step === 3 ? ['c'] : ['a', 'b', 'c'];
	if (step < 4) return { shown, target: null, feeders: { a: [], b: [] } };
	const target = elementOf('c', c.lane, step - 4);
	return { shown, target, feeders: { a: lanesForRowOfA(target.row), b: lanesForColOfB(target.col) } };
}

const list = (es: Element[]) => es.map((e) => `(${e.row}, ${e.col})`).join(' ');

export function describe(c: Config, step: number) {
	const l = c.lane;
	const g = groupId(l);
	const t = threadInGroup(l);
	const head = `Lane ${l}: groupID = ${l} >> 2 = ${g}, threadID_in_group = ${l} % 4 = ${t}.`;
	switch (step) {
		case 0:
			return `Step 0: one warp, 32 lanes, computes D = A x B + C for a 16 x 8 x 16 tile. Each color is one lane's fragment. ${head}`;
		case 1:
			return `Step 1, A (16 x 16, f16): every lane holds 8 elements in four 32-bit registers, two halves each. ${head} It holds ${list(elementsOf('a', l))} as a0 to a7: rows ${g} and ${g + 8}, columns ${2 * t}, ${2 * t + 1}, ${2 * t + 8} and ${2 * t + 9}.`;
		case 2:
			return `Step 2, B (16 x 8, f16): 4 elements per lane in two registers, all in column groupID = ${g}. Lane ${l} holds ${list(elementsOf('b', l))} as b0 to b3. In A and in B, the two halves of one register are neighbours along k.`;
		case 3:
			return `Step 3, C and D (16 x 8, f32): 4 elements per lane, one register each. Lane ${l} holds ${list(elementsOf('c', l))} as c0 to c3, the same rows and columns as its first four A elements. D comes back in the same layout, so the next mma can use it as C.`;
		default: {
			const i = step - 4;
			const v = viewAt(c, step);
			const e = v.target!;
			return `Step ${step}, d${i} of lane ${l} = D(${e.row}, ${e.col}) = C(${e.row}, ${e.col}) + sum over k of A(${e.row}, k) x B(k, ${e.col}), 16 products. Row ${e.row} of A sits in lanes ${v.feeders.a.join(', ')}; column ${e.col} of B in lanes ${v.feeders.b.join(', ')}. The tensor core gathers them across the warp; no lane does this on its own.`;
		}
	}
}

export function pickText(p: NonNullable<Config['pick']>) {
	const o = ownerOf(p.m, p.row, p.col);
	const name = p.m === 'c' ? 'C/D' : p.m.toUpperCase();
	return `${name}(${p.row}, ${p.col}) is held by lane ${o.lane} as ${elemName(p.m, o.i)}, in ${regText(p.m, o.i)}.`;
}
