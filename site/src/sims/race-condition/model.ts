/**
 * Threads that each add 1 to one counter in shared memory, `count += 1`.
 *
 * The compiler turns `count += 1` into three instructions: load the counter
 * into a register, add 1 to the register, store the register back. NVRTC
 * emits exactly that for kernels/ch22-profiling/cuda/bugs_kernels.cuh
 * (ld.shared.u32, add.s32, st.shared.u32; see generated/count_racy.ptx).
 * Between two of one thread's instructions, any other thread can run. The
 * reader picks the order. With atomicAdd the three become one instruction,
 * which the CUDA C++ Programming Guide says is "performed in one atomic
 * transaction", so no order can split it.
 *
 * The model runs one instruction per step and nothing else: no warps, no
 * memory latency, no caches. On a GPU the 32 threads of a warp run the same
 * instruction together, which is the "every thread loads first" order.
 */
export const THREAD_OPTIONS = [2, 3, 4] as const;

export const MODE_IDS = ['plain', 'atomic'] as const;
export type Mode = (typeof MODE_IDS)[number];
export const MODE_NAMES: Record<Mode, string> = {
	plain: 'count += 1 (load, add, store)',
	atomic: 'atomicAdd(&count, 1)',
};

export const PRESET_IDS = ['serial', 'lost-one', 'all-load', 'custom'] as const;
export type Preset = (typeof PRESET_IDS)[number];
export const PRESET_NAMES: Record<Preset, string> = {
	serial: 'One thread after another',
	'lost-one': 'Thread 1 loads before thread 0 stores',
	'all-load': 'Every thread loads first, like a warp',
	custom: 'Custom: you pick who runs next',
};

export type Op = 'load' | 'add' | 'store' | 'atomic';
export const OPS: Record<Mode, Op[]> = { plain: ['load', 'add', 'store'], atomic: ['atomic'] };
/** The instruction as the panels print it. */
export const OP_TEXT: Record<Op, string> = {
	load: 'r = count',
	add: 'r = r + 1',
	store: 'count = r',
	atomic: 'atomicAdd(&count, 1)',
};

export interface Config {
	threads: number;
	mode: Mode;
	preset: Preset;
	/** Thread picks for the custom preset, in order. */
	custom: number[];
}

export interface Event {
	thread: number;
	op: Op;
}

/** Thread picks for a preset: which thread runs next, before finished threads are skipped. */
export function picks(c: Config): number[] {
	const ids = Array.from({ length: c.threads }, (_, t) => t);
	const serial = (ts: number[]) => ts.flatMap((t) => [t, t, t]);
	switch (c.preset) {
		case 'serial':
			return serial(ids);
		case 'lost-one':
			// Thread 1 loads between thread 0's load and store; the rest run one after another.
			return [0, 1, 0, 0, 1, 1, ...serial(ids.slice(2))];
		case 'all-load':
			return [...ids, ...ids, ...ids];
		case 'custom':
			return c.custom;
	}
}

/** The instructions in the order they run. A pick of a finished thread is skipped. */
export function schedule(c: Config): Event[] {
	const ops = OPS[c.mode];
	const done = new Array(c.threads).fill(0);
	const out: Event[] = [];
	for (const t of picks(c)) {
		if (t < 0 || t >= c.threads || done[t] >= ops.length) continue;
		out.push({ thread: t, op: ops[done[t]++] });
	}
	return out;
}

export const lastStep = (c: Config) => schedule(c).length;
/** Every thread has run all its instructions. */
export const isComplete = (c: Config) => lastStep(c) === c.threads * OPS[c.mode].length;

export interface ThreadState {
	/** Instructions finished. */
	done: number;
	/** The thread's register; null before its first load. */
	reg: number | null;
	/** Value of count when this thread loaded it. */
	loaded: number | null;
}

export interface State {
	count: number;
	threads: ThreadState[];
	/** Updates lost so far. */
	lost: number;
	/** The instruction that ran to reach this step, with count before and after it. */
	event: (Event & { before: number; after: number; lost: number }) | null;
	/** Index into the schedule of every store that lost updates, up to this step. */
	lossySteps: number[];
}

export function stateAt(c: Config, step: number): State {
	const events = schedule(c);
	const s: State = {
		count: 0,
		threads: Array.from({ length: c.threads }, () => ({ done: 0, reg: null, loaded: null })),
		lost: 0,
		event: null,
		lossySteps: [],
	};
	const k = Math.max(0, Math.min(step, events.length));
	for (let i = 0; i < k; i++) {
		const e = events[i];
		const th = s.threads[e.thread];
		const before = s.count;
		let lost = 0;
		if (e.op === 'load') {
			th.reg = s.count;
			th.loaded = s.count;
		} else if (e.op === 'add') {
			th.reg = (th.reg ?? 0) + 1;
		} else if (e.op === 'store') {
			// A correct update would write count + 1. Writing loaded + 1 instead
			// drops every store that landed after this thread's load.
			lost = s.count - (th.loaded ?? 0);
			s.count = th.reg ?? 0;
		} else {
			th.reg = s.count;
			s.count += 1;
		}
		th.done++;
		s.lost += lost;
		if (lost > 0) s.lossySteps.push(i);
		s.event = { ...e, before, after: s.count, lost };
	}
	return s;
}

const plural = (n: number, w: string, ws = `${w}s`) => `${n} ${n === 1 ? w : ws}`;

export function describe(c: Config, step: number): string {
	const last = lastStep(c);
	const s = stateAt(c, step);
	if (step === 0 || !s.event) {
		const how = c.preset === 'custom' ? ' Pick which thread runs next with the buttons above the drawing.' : ' Predict what count holds at the end, then step.';
		return `${c.threads} threads each add 1 to count, which starts at 0. Expected at the end: ${c.threads}.${how}`;
	}
	const e = s.event;
	const t = `Thread ${e.thread}`;
	let what: string;
	if (e.op === 'load') what = `${t} loads count (${e.before}) into its register: r = ${e.before}.`;
	else if (e.op === 'add') what = `${t} adds 1 to its register: r = ${s.threads[e.thread].reg}. count is still ${s.count}.`;
	else if (e.op === 'store')
		what =
			e.lost > 0
				? `${t} stores r = ${e.after} over count = ${e.before}. It loaded ${s.threads[e.thread].loaded}, before ${plural(e.lost, 'other update')} landed, so ${e.lost === 1 ? 'that update is' : 'those updates are'} lost.`
				: `${t} stores r = ${e.after}. count goes from ${e.before} to ${e.after}.`;
	else what = `${t} runs atomicAdd: it reads ${e.before}, writes ${e.after}, and no other thread can run in between.`;
	let tail = '';
	if (step === last && isComplete(c)) {
		tail = s.lost > 0 ? ` Done. Expected ${c.threads}, count holds ${s.count}: ${plural(s.lost, 'update')} lost.` : ` Done. count holds ${s.count}, as expected: no update lost.`;
	}
	return `Step ${step}. ${what}${tail}`;
}

/** Code shown in the sim's panel. Lines are 1-based. */
export const CODE: Record<Mode, string> = {
	plain: `// count += 1, as three instructions
r = count;      // ld.shared.u32
r = r + 1;      // add.s32
count = r;      // st.shared.u32`,
	atomic: `// one indivisible read-modify-write
old = atomicAdd(&count, 1);  // atom.shared.add.u32`,
};

const LINE: Record<Op, number> = { load: 2, add: 3, store: 4, atomic: 2 };
export const lineOf = (op: Op) => LINE[op];
