/**
 * One warp of 32 lanes running an if/else. The warp issues one instruction
 * at a time; lanes not on the current path are masked off. The order shown
 * (if path, then else path) is one valid order: since Volta the scheduler may
 * interleave the two paths, but each instruction still runs for one path's lanes.
 */
export const WARP = 32;
export const BLOCK = 64;

export type CondId = 'parity' | 'lower' | 'data' | 'uniform';

export const CONDITIONS: Record<CondId, { label: string; expr: string; test: (tid: number) => boolean }> = {
	parity: { label: 'Even lanes', expr: 'threadIdx.x % 2 == 0', test: (tid) => tid % 2 === 0 },
	lower: { label: 'First 8 lanes of each warp', expr: 'threadIdx.x % 32 < 8', test: (tid) => tid % 32 < 8 },
	data: { label: 'Data: x[i] > 0', expr: 'v > 0.0f', test: (tid) => X[tid] > 0 },
	uniform: { label: 'Whole warps: threadIdx.x < 32', expr: 'threadIdx.x < 32', test: (tid) => tid < 32 },
};
export const COND_IDS = Object.keys(CONDITIONS) as CondId[];

/** Fixed input for the data condition: deterministic, with mixed signs in both warps. */
export const X = Array.from({ length: BLOCK }, (_, i) => Math.round(100 * Math.sin(i * 2.3 + 0.7)) / 100);

export interface Config {
	cond: CondId;
	/** Which warp of the 64-thread block: 0 is threads 0 to 31, 1 is threads 32 to 63. */
	warp: 0 | 1;
}

export type Path = 'all' | 'if' | 'else';
export interface Instr {
	line: number;
	path: Path;
	/** Lanes that execute this instruction. */
	active: boolean[];
}

export const codeFor = (cond: CondId) => `int i = blockIdx.x * blockDim.x + threadIdx.x;
float v = x[i];
if (${CONDITIONS[cond].expr}) {
    v = v * 2.0f;
    v = v + 1.0f;
} else {
    v = v * 0.5f;
}
out[i] = v;`;

export const tidOf = (c: Config, lane: number) => c.warp * WARP + lane;
export const takesIf = (c: Config) => Array.from({ length: WARP }, (_, lane) => CONDITIONS[c.cond].test(tidOf(c, lane)));

/** The instructions the warp issues, in order. A path no lane takes is skipped. */
export function program(c: Config): Instr[] {
	const t = takesIf(c);
	const all = t.map(() => true);
	const f = t.map((x) => !x);
	const out: Instr[] = [
		{ line: 1, path: 'all', active: all },
		{ line: 2, path: 'all', active: all },
		{ line: 3, path: 'all', active: all },
	];
	if (t.some(Boolean)) out.push({ line: 4, path: 'if', active: t }, { line: 5, path: 'if', active: t });
	if (f.some(Boolean)) out.push({ line: 7, path: 'else', active: f });
	out.push({ line: 9, path: 'all', active: all });
	return out;
}

export const lastStep = (c: Config) => program(c).length;
const count = (a: boolean[]) => a.filter(Boolean).length;

export interface State {
	step: number;
	instr: Instr | null;
	/** Branch outcome per lane, known once line 3 has run. */
	outcome: boolean[] | null;
	/** Instructions issued so far and the lane slots they used. */
	issued: Instr[];
	usedSlots: number;
}

export function stateAt(c: Config, step: number): State {
	const prog = program(c);
	const s = Math.max(0, Math.min(step, prog.length));
	const issued = prog.slice(0, s);
	return {
		step: s,
		instr: s ? prog[s - 1] : null,
		outcome: s >= 3 ? takesIf(c) : null,
		issued,
		usedSlots: issued.reduce((n, x) => n + count(x.active), 0),
	};
}

/** Fraction of lane slots doing useful work over the whole run. */
export function efficiency(c: Config) {
	const prog = program(c);
	return prog.reduce((n, x) => n + count(x.active), 0) / (prog.length * WARP);
}

const pct = (f: number) => `${(100 * f).toFixed(1)}%`;

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	const t = takesIf(c);
	const nIf = count(t);
	const nElse = WARP - nIf;
	const first = c.warp * WARP;
	if (!st.instr) {
		return `Warp ${c.warp} holds threads ${first} to ${first + WARP - 1} of the block, one per lane. The branch is if (${CONDITIONS[c.cond].expr}). Predict: how many lanes will sit idle during the if body? Then press Next.`;
	}
	const { line, path, active } = st.instr;
	let text: string;
	if (line === 1) text = 'Line 1: all 32 lanes compute their own i. One instruction, 32 results.';
	else if (line === 2) text = 'Line 2: all 32 lanes load their v = x[i].';
	else if (line === 3) {
		text = `Line 3: all 32 lanes evaluate the condition. ${nIf} are true, ${nElse} are false.`;
		text +=
			nIf && nElse
				? ' The lanes disagree, so the warp diverges: it must run both paths, one after the other.'
				: ` Every lane agrees, so the warp does not diverge and the ${nIf ? 'else' : 'if'} path is skipped.`;
	} else if (path === 'if') {
		text = `Line ${line}, if path: ${count(active)} lanes run it; ${WARP - count(active)} lanes are masked off and do nothing for this instruction.`;
	} else if (path === 'else') {
		text = `Line 7, else path: ${count(active)} lanes run it; the ${WARP - count(active)} lanes that took the if path are masked off.`;
	} else {
		text = nIf && nElse ? 'Line 9: the paths reconverge and all 32 lanes store out[i].' : 'Line 9: all 32 lanes store out[i].';
	}
	if (st.step === lastStep(c)) {
		const total = st.issued.length * WARP;
		text += ` Done: the warp issued ${st.issued.length} instructions with room for ${total} lane results and used ${st.usedSlots} of them (${pct(st.usedSlots / total)}).`;
	}
	return text;
}
