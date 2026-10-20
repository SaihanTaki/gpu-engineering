/**
 * Hillis-Steele and Blelloch prefix sums on one block of 8 or 16 integers, phase by phase.
 *
 * Hillis and Steele, "Data Parallel Algorithms", Communications of the ACM 29(12), 1986:
 * at step k every element adds the one 2^k places to its left. log2(n) steps,
 * n log2(n) - (n - 1) additions, inclusive result.
 * Blelloch, "Prefix Sums and Their Applications", CMU-CS-90-190, 1990: an up-sweep
 * reduction tree, clear the last element, a down-sweep. 2 log2(n) addition levels,
 * 2(n - 1) additions, exclusive result. Indices follow Harris, Sengupta and Owens,
 * GPU Gems 3, chapter 39.
 *
 * The other result type is a shift by one with no additions: Hillis-Steele shifts
 * right and puts 0 first, Blelloch shifts left and puts the saved total last.
 * Both match kernels/ch14-scan.
 */
export const SIZES = [8, 16] as const;
export type Size = (typeof SIZES)[number];
export const MODES = ['inclusive', 'exclusive'] as const;
export type Mode = (typeof MODES)[number];

export const INPUTS = {
	// The first 8 are Blelloch's example in "Prefix Sums and Their Applications".
	mixed: [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 1, 0, 3, 4, 2, 1],
	ones: new Array(16).fill(1) as number[],
};
export type InputId = keyof typeof INPUTS;
export const INPUT_NAMES: Record<InputId, string> = { mixed: 'Mixed digits', ones: 'All ones' };

export const ALGO_IDS = ['hs', 'blelloch'] as const;
export type AlgoId = (typeof ALGO_IDS)[number];
export const ALGO_NAMES: Record<AlgoId, string> = { hs: 'Hillis-Steele', blelloch: 'Blelloch' };

export interface Config {
	n: Size;
	mode: Mode;
	input: InputId;
}

/** add: after[to] = before[to] + before[from]. copy: after[to] = before[from]. */
export interface Op {
	kind: 'add' | 'copy';
	from: number;
	to: number;
}

export type PhaseKind = 'level' | 'up' | 'clear' | 'down' | 'shift';

export interface Phase {
	kind: PhaseKind;
	/** Distance between the two elements an operation joins; 0 for clear and shift. */
	offset: number;
	/** Short row label. */
	label: string;
	ops: Op[];
	/** Cells written in this phase. */
	changed: number[];
	/** Values after this phase. */
	values: number[];
	adds: number;
}

export const log2 = (n: number) => Math.round(Math.log2(n));
export const inputOf = (c: Config) => INPUTS[c.input].slice(0, c.n);

function apply(before: number[], ops: Op[]): number[] {
	const after = before.slice();
	for (const op of ops) after[op.to] = op.kind === 'add' ? before[op.to] + before[op.from] : before[op.from];
	return after;
}

function phase(kind: PhaseKind, offset: number, label: string, before: number[], ops: Op[], extra?: (v: number[]) => number[]): Phase {
	const values = apply(before, ops);
	const changed = new Set(ops.map((o) => o.to));
	if (extra) for (const i of extra(values)) changed.add(i);
	return { kind, offset, label, ops, changed: [...changed].sort((a, b) => a - b), values, adds: ops.filter((o) => o.kind === 'add').length };
}

function hillisSteele(x: number[], mode: Mode): Phase[] {
	const n = x.length;
	const out: Phase[] = [];
	let v = x;
	for (let o = 1; o < n; o *= 2) {
		const ops: Op[] = [];
		for (let i = o; i < n; i++) ops.push({ kind: 'add', from: i - o, to: i });
		const p = phase('level', o, `+${o}`, v, ops);
		out.push(p);
		v = p.values;
	}
	if (mode === 'exclusive') {
		const ops: Op[] = [];
		for (let i = 1; i < n; i++) ops.push({ kind: 'copy', from: i - 1, to: i });
		out.push(
			phase('shift', 0, 'shift', v, ops, (vals) => {
				vals[0] = 0;
				return [0];
			}),
		);
	}
	return out;
}

function blelloch(x: number[], mode: Mode): Phase[] {
	const n = x.length;
	const out: Phase[] = [];
	let v = x;
	let level = 0;
	for (let o = 1; o < n; o *= 2) {
		const ops: Op[] = [];
		for (let t = 0; t < n / (2 * o); t++) ops.push({ kind: 'add', from: o * (2 * t + 1) - 1, to: o * (2 * t + 2) - 1 });
		const p = phase('up', o, `up ${++level}`, v, ops);
		out.push(p);
		v = p.values;
	}
	const total = v[n - 1];
	const clear = phase('clear', 0, 'clear', v, [], (vals) => {
		vals[n - 1] = 0;
		return [n - 1];
	});
	out.push(clear);
	v = clear.values;
	level = 0;
	for (let o = n / 2; o >= 1; o /= 2) {
		const ops: Op[] = [];
		for (let t = 0; t < n / (2 * o); t++) {
			const a = o * (2 * t + 1) - 1;
			const b = o * (2 * t + 2) - 1;
			ops.push({ kind: 'copy', from: b, to: a }, { kind: 'add', from: a, to: b });
		}
		const p = phase('down', o, `down ${++level}`, v, ops);
		out.push(p);
		v = p.values;
	}
	if (mode === 'inclusive') {
		const ops: Op[] = [];
		for (let i = 0; i < n - 1; i++) ops.push({ kind: 'copy', from: i + 1, to: i });
		out.push(
			phase('shift', 0, 'shift', v, ops, (vals) => {
				vals[n - 1] = total;
				return [n - 1];
			}),
		);
	}
	return out;
}

export function phases(c: Config, algo: AlgoId): Phase[] {
	const x = inputOf(c);
	return algo === 'hs' ? hillisSteele(x, c.mode) : blelloch(x, c.mode);
}

/** Step 0 is the input; step s shows each algorithm after its s-th phase, or its result if it has fewer. */
export const lastStep = (c: Config) => Math.max(...ALGO_IDS.map((a) => phases(c, a).length));

export interface Counts {
	phases: number;
	/** Phases that contain additions: the parallel steps that do work. */
	addLevels: number;
	adds: number;
	done: boolean;
}

export function countsAt(c: Config, algo: AlgoId, step: number): Counts {
	const ps = phases(c, algo);
	const done = ps.slice(0, Math.min(step, ps.length));
	return {
		phases: done.length,
		addLevels: done.filter((p) => p.adds > 0).length,
		adds: done.reduce((s, p) => s + p.adds, 0),
		done: step >= ps.length,
	};
}

/** Values shown at a step: the input at step 0, the last phase once an algorithm is done. */
export function valuesAt(c: Config, algo: AlgoId, step: number): number[] {
	const ps = phases(c, algo);
	if (step === 0) return inputOf(c);
	return ps[Math.min(step, ps.length) - 1].values;
}

export function reference(c: Config): number[] {
	let run = 0;
	return inputOf(c).map((x) => {
		const before = run;
		run += x;
		return c.mode === 'inclusive' ? run : before;
	});
}

/** Closed forms for a block of n = 2^k elements, ignoring the zero-addition shift and clear phases. */
export const work = {
	sequential: (n: number) => n - 1,
	hs: (n: number) => n * log2(n) - (n - 1),
	blelloch: (n: number) => 2 * (n - 1),
};
export const depth = {
	sequential: (n: number) => n - 1,
	hs: (n: number) => log2(n),
	blelloch: (n: number) => 2 * log2(n),
};

export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
const plural = (n: number, w: string) => `${n} ${n === 1 ? w : `${w}s`}`;

export function describePhase(p: Phase, algo: AlgoId, total: number): string {
	switch (p.kind) {
		case 'level':
			return `every element from index ${p.offset} on adds the element ${p.offset} ${p.offset === 1 ? 'place' : 'places'} to its left: ${plural(p.adds, 'addition')}`;
		case 'up':
			return `up-sweep: ${plural(p.adds, 'element')} ${p.adds === 1 ? 'adds' : 'each add'} the partial sum ${p.offset} ${p.offset === 1 ? 'place' : 'places'} to the left: ${plural(p.adds, 'addition')}`;
		case 'clear':
			return `saves the total, ${total}, and sets the last element to 0: no additions`;
		case 'down':
			return `down-sweep: in ${plural(p.adds, 'pair')} ${p.offset} apart, the left element takes the right one's value and the right one adds the old left value: ${plural(p.adds, 'addition')}`;
		case 'shift':
			return algo === 'hs'
				? 'shifts the inclusive result right by one and puts 0 first, which makes it exclusive: no additions'
				: `shifts the exclusive result left by one and puts the saved total, ${total}, last, which makes it inclusive: no additions`;
	}
}

export function describeAlgo(c: Config, algo: AlgoId, step: number): string {
	const ps = phases(c, algo);
	const name = ALGO_NAMES[algo];
	const total = inputOf(c).reduce((s, x) => s + x, 0);
	const k = countsAt(c, algo, step);
	const tally = `${plural(k.adds, 'addition')} in ${plural(k.addLevels, 'addition level')} so far`;
	if (step > ps.length) return `${name} is done: ${tally.replace(' so far', '')}.`;
	return `${name}, ${ps[step - 1].label}: ${describePhase(ps[step - 1], algo, total)}. ${tally}.`;
}

export function describe(c: Config, step: number): string {
	const x = inputOf(c);
	if (step === 0) {
		return `Input: ${x.join(', ')}. Both algorithms compute the ${c.mode} scan of these ${c.n} values. A sequential loop needs ${work.sequential(c.n)} additions, one after another. Guess how many additions each parallel algorithm needs, then step.`;
	}
	const parts = ALGO_IDS.map((a) => describeAlgo(c, a, step));
	if (step === lastStep(c)) {
		parts.push(`Result: ${reference(c).join(', ')}.`);
	}
	return parts.join(' ');
}

/** Source shown in the code panel; one block, simplified from kernels/ch14-scan/cuda/scan_kernels.cuh. */
export const CODE: Record<AlgoId, string> = {
	hs: `s[tid] = in[i];
for (int offset = 1; offset < n; offset *= 2) {
    int left = tid >= offset ? s[tid - offset] : 0;
    __syncthreads();
    if (tid >= offset) s[tid] += left;
    __syncthreads();
}
out[i] = INCLUSIVE ? s[tid]
                   : (tid > 0 ? s[tid - 1] : 0);`,
	blelloch: `// a = offset * (2 * tid + 1) - 1, b = a + offset
s[k] = in[base + k];  // two per thread
int offset = 1;
for (int d = n / 2; d > 0; d /= 2) {
    __syncthreads();
    if (tid < d) s[b] += s[a];
    offset *= 2;
}
if (tid == 0) { total = s[n - 1]; s[n - 1] = 0; }
for (int d = 1; d < n; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (tid < d) { int t = s[a]; s[a] = s[b]; s[b] += t; }
}
__syncthreads();
out[k] = !INCLUSIVE ? s[k]
                    : (k + 1 < n ? s[k + 1] : total);`,
};

const LINES: Record<AlgoId, Partial<Record<PhaseKind | 'input', number[]>>> = {
	hs: { input: [1], level: [3, 5], shift: [8, 9] },
	blelloch: { input: [2], up: [6], clear: [9], down: [13], shift: [16, 17] },
};

/** 1-based code lines active at a step for one algorithm. */
export function linesAt(c: Config, algo: AlgoId, step: number): number[] {
	if (step === 0) return LINES[algo].input ?? [];
	const ps = phases(c, algo);
	if (step > ps.length) return [];
	return LINES[algo][ps[step - 1].kind] ?? [];
}
