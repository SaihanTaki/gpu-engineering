export type GridMode = 'ceil' | 'floor';
export type Phase = 'index' | 'guard' | 'write';

export interface Config {
	n: number;
	blockDim: number;
	/** 'ceil' is (n + blockDim - 1) / blockDim; 'floor' is the common bug n / blockDim. */
	gridMode: GridMode;
	/** Whether the kernel has the `if (i < n)` bounds check. */
	guard: boolean;
}

export interface Thread {
	block: number;
	thread: number;
	i: number;
	inBounds: boolean;
	/** True if this thread executes the store, in bounds or not. */
	writes: boolean;
}

export interface State {
	step: number;
	/** Block running at this step, or null before launch. */
	block: number | null;
	phase: Phase | null;
	/** For each element of out[], the block that wrote it, or null. */
	writtenBy: (number | null)[];
	/** Indices >= n that were written so far (only without the guard). */
	outOfBounds: number[];
	/** 1-based lines of the code variant to highlight. */
	lines: number[];
}

export const gridDim = ({ n, blockDim, gridMode }: Config) =>
	gridMode === 'ceil' ? Math.ceil(n / blockDim) : Math.floor(n / blockDim);

export function threadOf(c: Config, block: number, thread: number): Thread {
	const i = block * c.blockDim + thread;
	const inBounds = i < c.n;
	return { block, thread, i, inBounds, writes: inBounds || !c.guard };
}

export function threads(c: Config): Thread[] {
	const out: Thread[] = [];
	for (let b = 0; b < gridDim(c); b++)
		for (let t = 0; t < c.blockDim; t++) out.push(threadOf(c, b, t));
	return out;
}

const phasesFor = (c: Config): Phase[] => (c.guard ? ['index', 'guard', 'write'] : ['index', 'write']);

/** Line numbers match codeFor() below. */
const LINES: Record<'guard' | 'noGuard', Record<Phase, number[]>> = {
	guard: { index: [2], guard: [3], write: [4] },
	noGuard: { index: [2], guard: [], write: [3] },
};

export const lastStep = (c: Config) => gridDim(c) * phasesFor(c).length;

export function stateAt(c: Config, step: number): State {
	const phases = phasesFor(c);
	const blocks = gridDim(c);
	const s = Math.min(Math.max(0, step), lastStep(c));
	const writtenBy: (number | null)[] = Array(c.n).fill(null);
	const outOfBounds: number[] = [];

	if (s === 0) return { step: 0, block: null, phase: null, writtenBy, outOfBounds, lines: [] };

	const block = Math.floor((s - 1) / phases.length);
	const phase = phases[(s - 1) % phases.length];
	const finished = phase === 'write' ? block + 1 : block;

	for (let b = 0; b < Math.min(finished, blocks); b++) {
		for (let t = 0; t < c.blockDim; t++) {
			const th = threadOf(c, b, t);
			if (!th.writes) continue;
			if (th.inBounds) writtenBy[th.i] = b;
			else outOfBounds.push(th.i);
		}
	}
	const lines = LINES[c.guard ? 'guard' : 'noGuard'][phase];
	return { step: s, block, phase, writtenBy, outOfBounds, lines };
}

/** Elements no thread maps to: only happens when the grid is too small. */
export function uncovered(c: Config): number[] {
	const covered = gridDim(c) * c.blockDim;
	return Array.from({ length: Math.max(0, c.n - covered) }, (_, k) => covered + k);
}

export function describe(c: Config, step: number): string {
	const blocks = gridDim(c);
	const total = blocks * c.blockDim;
	if (step === 0) {
		if (blocks === 0) return `gridDim.x = ${c.n} / ${c.blockDim} = 0, so no blocks launch and nothing is written.`;
		return `Before launch: ${blocks} block${blocks === 1 ? '' : 's'} of ${c.blockDim} threads, ${total} threads in total.`;
	}
	const st = stateAt(c, step);
	const b = st.block!;
	const first = b * c.blockDim;
	const last = first + c.blockDim - 1;
	const inBounds = Math.max(0, Math.min(last, c.n - 1) - first + 1);
	const idle = c.blockDim - inBounds;
	let text: string;
	if (st.phase === 'index') {
		text = `Block ${b}: every thread computes i = ${b} * ${c.blockDim} + threadIdx.x, giving i = ${first} to ${last}.`;
	} else if (st.phase === 'guard') {
		text = idle
			? `Block ${b}: ${inBounds} threads pass i < ${c.n}; ${idle} thread${idle === 1 ? '' : 's'} with i >= ${c.n} skip the store.`
			: `Block ${b}: all ${c.blockDim} threads pass i < ${c.n}.`;
	} else {
		const oob = c.guard ? 0 : idle;
		text = inBounds
			? `Block ${b} writes out[${first}] to out[${first + inBounds - 1}].`
			: `Block ${b} writes nothing inside out[].`;
		if (oob) text += ` ${oob} thread${oob === 1 ? '' : 's'} write past the end of out[]: undefined behavior.`;
	}
	if (step === lastStep(c)) {
		const missing = uncovered(c).length;
		text += missing
			? ` Done. ${missing} element${missing === 1 ? ' was' : 's were'} never written because no thread maps to ${missing === 1 ? 'it' : 'them'}.`
			: ` Done. Every element of out[] was written once.`;
	}
	return text;
}

export function codeFor(guard: boolean): string {
	return guard
		? `__global__ void whoami(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}`
		: `__global__ void whoami(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;
}`;
}
