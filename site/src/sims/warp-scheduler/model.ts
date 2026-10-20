/**
 * One warp scheduler issuing at most one instruction per cycle from the
 * warps assigned to it. Each warp loops ITERATIONS times over one load and
 * `alu` arithmetic instructions that all need the loaded value.
 * Left out: dependent arithmetic latency, instruction-level parallelism,
 * the store and loop overhead, and memory bandwidth limits.
 */
export const ITERATIONS = 4;
export const LIMITS = {
	warps: { min: 1, max: 8 },
	latency: { min: 4, max: 40 },
	alu: { min: 1, max: 6 },
} as const;

export interface Config {
	warps: number;
	/** Cycles from issuing a load until an instruction that uses its result can issue. */
	latency: number;
	/** Arithmetic instructions per load. */
	alu: number;
}

export type Kind = 'load' | 'alu';
/** What a warp is doing in one cycle. */
export type WarpState = Kind | 'waiting' | 'ready' | 'done';

export interface Cycle {
	cycle: number;
	/** Warp that issued this cycle, or null if the scheduler was idle. */
	issued: number | null;
	kind: Kind | null;
	/** Iteration (0-based) and position within it of the issued instruction. */
	iteration: number;
	slot: number;
	warps: WarpState[];
	/** Cycle at which each warp can next issue, as seen at the start of this cycle. */
	readyAt: number[];
}

export const programLength = (c: Config) => ITERATIONS * (1 + c.alu);
const kindAt = (c: Config, pc: number): Kind => (pc % (1 + c.alu) === 0 ? 'load' : 'alu');

/** Smallest warp count that can issue every cycle once the loop is running. */
export const warpsToHide = (c: Config) => Math.ceil((c.latency + c.alu) / (1 + c.alu));

const cache = new Map<string, Cycle[]>();

/** Runs the whole kernel. Greedy then oldest: keep issuing from the last warp while it is ready, else pick the warp ready longest. */
export function simulate(c: Config): Cycle[] {
	const key = `${c.warps}/${c.latency}/${c.alu}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const len = programLength(c);
	const pc = Array(c.warps).fill(0);
	const readyAt = Array(c.warps).fill(0);
	let last: number | null = null;
	const trace: Cycle[] = [];
	for (let t = 0; pc.some((p) => p < len); t++) {
		const ready = (w: number) => pc[w] < len && readyAt[w] <= t;
		let pick: number | null = last !== null && ready(last) ? last : null;
		// Oldest: the ready warp that has waited longest, lowest number on ties.
		if (pick === null) for (let w = 0; w < c.warps; w++) if (ready(w) && (pick === null || readyAt[w] < readyAt[pick])) pick = w;
		const warps: WarpState[] = pc.map((p, w) =>
			p >= len ? 'done' : w === pick ? kindAt(c, p) : readyAt[w] > t ? 'waiting' : 'ready',
		);
		const entry: Cycle = { cycle: t, issued: pick, kind: null, iteration: 0, slot: 0, warps, readyAt: [...readyAt] };
		if (pick !== null) {
			const p = pc[pick];
			entry.kind = kindAt(c, p);
			entry.iteration = Math.floor(p / (1 + c.alu));
			entry.slot = p % (1 + c.alu);
			readyAt[pick] = t + (entry.kind === 'load' ? c.latency : 1);
			pc[pick] = p + 1;
			last = pick;
		}
		trace.push(entry);
	}
	cache.set(key, trace);
	return trace;
}

export const lastStep = (c: Config) => simulate(c).length;

export interface Totals {
	cycles: number;
	issued: number;
	idle: number;
	/** Fraction of cycles in which the scheduler issued an instruction. */
	busy: number;
}

/** Totals after `step` cycles have run. */
export function totalsAt(c: Config, step: number): Totals {
	const done = simulate(c).slice(0, Math.max(0, Math.min(step, lastStep(c))));
	const issued = done.filter((x) => x.issued !== null).length;
	return { cycles: done.length, issued, idle: done.length - issued, busy: done.length ? issued / done.length : 0 };
}

/** 1-based lines of CODE for the instruction issued in this step. */
export function linesAt(c: Config, step: number): number[] {
	if (step < 1) return [];
	const cy = simulate(c)[Math.min(step, lastStep(c)) - 1];
	return cy.kind === 'load' ? [5] : cy.kind === 'alu' ? [6] : [];
}

export const CODE = `__global__ void work(const float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < ${ITERATIONS}; k++) {
        float x = a[k * n + i];  // load: x arrives later
        acc += f(x);             // arithmetic that needs x
    }
    out[i] = acc;
}`;

const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
const pct = (f: number) => `${Math.round(100 * f)}%`;
const list = (ws: number[]) => (ws.length === 1 ? `warp ${ws[0]}` : `warps ${ws.slice(0, -1).join(', ')} and ${ws.at(-1)}`);

export function describe(c: Config, step: number): string {
	const trace = simulate(c);
	const need = warpsToHide(c);
	if (step <= 0) {
		return `${c.warps} warp${c.warps === 1 ? '' : 's'} share this scheduler. Each runs ${ITERATIONS} iterations of one load and ${c.alu} arithmetic instruction${c.alu === 1 ? '' : 's'} that need the loaded value. A load takes ${c.latency} cycles to return. Predict: in how many cycles will the scheduler have nothing to issue? Then press Next.`;
	}
	const s = Math.min(step, trace.length);
	const cy = trace[s - 1];
	const waiting = cy.warps.flatMap((st, w) => (st === 'waiting' ? [w] : []));
	const ready = cy.warps.flatMap((st, w) => (st === 'ready' ? [w] : []));
	let text: string;
	if (cy.issued === null) {
		const next = Math.min(...cy.readyAt.filter((_, w) => cy.warps[w] === 'waiting'));
		text = `Cycle ${cy.cycle}: no warp is ready, so the scheduler issues nothing. ${waiting.length ? `${cap(list(waiting))} wait${waiting.length === 1 ? 's' : ''} on memory; the first result arrives at cycle ${next}.` : ''}`;
	} else if (cy.kind === 'load') {
		text = `Cycle ${cy.cycle}: warp ${cy.issued} issues the load of iteration ${cy.iteration + 1}. Its next instruction needs x, so it cannot issue again before cycle ${cy.cycle + c.latency}.`;
	} else {
		text = `Cycle ${cy.cycle}: warp ${cy.issued} issues arithmetic instruction ${cy.slot} of ${c.alu} in iteration ${cy.iteration + 1}; its load has returned.`;
	}
	if (cy.issued !== null && waiting.length) text += ` Waiting on memory: ${list(waiting)}.`;
	if (ready.length) text += ` Ready but not picked: ${list(ready)}.`;
	if (s === trace.length) {
		const t = totalsAt(c, s);
		text += ` Done: ${t.issued} instructions in ${t.cycles} cycles, ${t.idle} idle, so the scheduler was busy ${pct(t.busy)} of the time.`;
		text +=
			c.warps >= need
				? ` With ${need} or more warps it never idles once the loop is running.`
				: ` It needs about ${need} warps to keep issuing every cycle; try more warps, or more arithmetic per load.`;
	}
	return text;
}
