/**
 * A race on one job of `work` units. Each unit takes a fixed number of ticks
 * on one worker. The serial units run one after another on a single worker;
 * the parallel units are shared out in rounds, one unit per worker per round.
 * The numbers are illustrative, not measurements of any real hardware.
 */
export interface Config {
	/** Parallel fraction in percent, 0 to 100. */
	p: number;
	/** Work units in the job; a multiple of 100 so the split is exact. */
	work: number;
	cpuWorkers: number;
	gpuWorkers: number;
}

export const CPU_TICKS_PER_UNIT = 1;
export const GPU_TICKS_PER_UNIT = 4;
export const STEPS = 40;
export const lastStep = STEPS;

export type LaneId = 'cpu' | 'gpu' | 'split';

export interface Lane {
	id: LaneId;
	serialUnits: number;
	parallelUnits: number;
	/** Ticks per unit for the serial part and for the parallel part. */
	serialCost: number;
	parallelCost: number;
	/** Workers sharing the parallel part. */
	workers: number;
	serialEnd: number;
	finish: number;
}

export type Phase = 'waiting' | 'serial' | 'parallel' | 'done';

export interface LaneState {
	phase: Phase;
	serialDone: number;
	parallelDone: number;
	/** Fraction of the whole job finished, 0 to 1. */
	fraction: number;
	/** Workers doing a unit right now. */
	busy: number;
}

/** Amdahl's law: speedup = 1 / ((1 - p) + p / n), with p as a fraction. */
export const amdahl = (p: number, n: number) => 1 / (1 - p + p / n);

export function split(c: Config) {
	const parallel = Math.round((c.work * c.p) / 100);
	return { serial: c.work - parallel, parallel };
}

function lane(id: LaneId, serialUnits: number, parallelUnits: number, serialCost: number, parallelCost: number, workers: number): Lane {
	const serialEnd = serialUnits * serialCost;
	const finish = serialEnd + Math.ceil(parallelUnits / workers) * parallelCost;
	return { id, serialUnits, parallelUnits, serialCost, parallelCost, workers, serialEnd, finish };
}

export function lanes(c: Config): Lane[] {
	const { serial, parallel } = split(c);
	return [
		lane('cpu', serial, parallel, CPU_TICKS_PER_UNIT, CPU_TICKS_PER_UNIT, c.cpuWorkers),
		lane('gpu', serial, parallel, GPU_TICKS_PER_UNIT, GPU_TICKS_PER_UNIT, c.gpuWorkers),
		// The hybrid: serial part on one CPU worker, parallel part on the GPU workers.
		lane('split', serial, parallel, CPU_TICKS_PER_UNIT, GPU_TICKS_PER_UNIT, c.gpuWorkers),
	];
}

export const laneById = (c: Config, id: LaneId) => lanes(c).find((l) => l.id === id)!;

/** The race ends when the slowest of the three lanes finishes. */
export const endTime = (c: Config) => Math.max(...lanes(c).map((l) => l.finish));

export function timeAt(c: Config, step: number) {
	const s = Math.min(Math.max(0, Math.round(step)), STEPS);
	return s === STEPS ? endTime(c) : (endTime(c) * s) / STEPS;
}

export function laneAt(l: Lane, t: number): LaneState {
	const total = l.serialUnits + l.parallelUnits;
	const make = (phase: Phase, serialDone: number, parallelDone: number, busy: number): LaneState => ({
		phase,
		serialDone,
		parallelDone,
		fraction: total ? (serialDone + parallelDone) / total : 1,
		busy,
	});
	if (t >= l.finish) return make('done', l.serialUnits, l.parallelUnits, 0);
	if (t <= 0) return make('waiting', 0, 0, 0);
	if (t < l.serialEnd) return make('serial', Math.floor(t / l.serialCost), 0, 1);
	const rounds = Math.floor((t - l.serialEnd) / l.parallelCost);
	const parallelDone = Math.min(l.parallelUnits, rounds * l.workers);
	return make('parallel', l.serialUnits, parallelDone, Math.min(l.workers, l.parallelUnits - parallelDone));
}

/** Time for the whole job on one worker of the lane's own kind. */
export const oneWorkerTime = (l: Lane) => l.serialUnits * l.serialCost + l.parallelUnits * l.parallelCost;

/** Speedup the model actually reaches, which rounds the last round up to a whole round. */
export const modelSpeedup = (l: Lane) => oneWorkerTime(l) / l.finish;

export function winner(c: Config): 'cpu' | 'gpu' | 'tie' {
	const cpu = laneById(c, 'cpu').finish;
	const gpu = laneById(c, 'gpu').finish;
	return cpu === gpu ? 'tie' : cpu < gpu ? 'cpu' : 'gpu';
}

export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
const pct = (f: number) => `${Math.floor(f * 100)}%`;

export const LANE_NAMES: Record<LaneId, string> = {
	cpu: 'CPU',
	gpu: 'GPU',
	split: 'CPU serial + GPU parallel',
};

/** Short phase text, shared by the SVG and the status line. */
export function phaseText(l: Lane, s: LaneState) {
	switch (s.phase) {
		case 'waiting':
			return 'not started';
		case 'serial':
			return 'serial, 1 worker busy';
		case 'parallel':
			return `parallel, ${s.busy} of ${l.workers} busy`;
		case 'done':
			return `done at t = ${fmt(l.finish)}`;
	}
}

export function describe(c: Config, step: number): string {
	const s = Math.min(Math.max(0, step), STEPS);
	const [cpu, gpu, both] = lanes(c);
	if (s === 0) {
		return (
			`A job of ${fmt(c.work)} units, ${c.p}% parallel. ` +
			`The CPU has ${c.cpuWorkers} workers at ${CPU_TICKS_PER_UNIT} tick per unit; ` +
			`the GPU has ${c.gpuWorkers} workers at ${GPU_TICKS_PER_UNIT} ticks per unit. ` +
			`Predict which finishes first, then press Next or Play.`
		);
	}
	const t = timeAt(c, s);
	const part = (l: Lane) => {
		const st = laneAt(l, t);
		return `${LANE_NAMES[l.id]} ${pct(st.fraction)}, ${phaseText(l, st)}.`;
	};
	let text = `t = ${fmt(t)} tick${fmt(t) === '1' ? '' : 's'}. ${part(cpu)} ${part(gpu)} ${part(both)}`;
	if (s === STEPS) {
		const w = winner(c);
		text +=
			w === 'tie'
				? ` The CPU and the GPU tie at ${fmt(cpu.finish)} ticks.`
				: w === 'cpu'
					? ` The CPU wins, ${fmt(cpu.finish)} ticks against ${fmt(gpu.finish)}.`
					: ` The GPU wins, ${fmt(gpu.finish)} ticks against ${fmt(cpu.finish)}.`;
		text += ` Running the serial part on the CPU and the parallel part on the GPU takes ${fmt(both.finish)}.`;
	}
	return text;
}
