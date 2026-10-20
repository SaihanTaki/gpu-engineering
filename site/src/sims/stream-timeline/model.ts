import type { GpuSpec } from '@sim-core/gpu-specs';

/**
 * A chunked copy-in, compute, copy-out pipeline spread over CUDA streams.
 *
 * Chunk c goes to stream c % streams, and the host issues its three operations
 * (H2D copy, kernel, D2H copy) in that order, chunk after chunk. The model then
 * places each operation on an engine:
 *   - one compute engine, running one kernel at a time (each chunk's kernel is
 *     assumed to fill the GPU),
 *   - one copy engine shared by both directions, or two (one per direction).
 * An operation starts when the previous operation in its stream has finished
 * and its engine is free. When several are ready, the earliest-ready runs first,
 * then the earliest issued.
 *
 * Copies run at the per-direction PCIe peak from gpu-specs.ts, a lower bound.
 * Model assumptions, because NVIDIA publishes no number for them:
 *   - every operation pays a fixed cost on its engine, picked by the reader,
 *   - a copy from pageable memory pays a staging copy at the PCIe rate (as in
 *     chapter 10) and overlaps with nothing, so a pageable run is fully serial.
 */
export const MIB = 1024 * 1024;
export const SIZES_MIB = [16, 64, 256];
export const STREAM_COUNTS = [1, 2, 3, 4];
export const CHUNK_COUNTS = [1, 2, 4, 8, 16];
/** Kernel time per chunk, as a multiple of that chunk's H2D copy time. */
export const KERNEL_RATIOS = [0.25, 0.5, 1, 2];
/** Assumed fixed cost of one operation, µs. */
export const OVERHEAD_US = [0, 5, 20];

export const DEFAULTS = { streams: 2, chunks: 8, mib: 64, kernelRatio: 1, pinned: true, engines: 2, overheadUs: 5 } as const;

export type Kind = 'h2d' | 'kernel' | 'd2h';
export type Engine = 'h2d' | 'd2h' | 'copy' | 'compute';

export interface Params {
	gpu: GpuSpec;
	streams: number;
	chunks: number;
	mib: number;
	kernelRatio: number;
	pinned: boolean;
	engines: 1 | 2;
	overheadUs: number;
}

export interface Op {
	kind: Kind;
	chunk: number;
	stream: number;
	engine: Engine;
	/** Position in host issue order, 0 first. */
	issue: number;
	bytes: number;
	start: number;
	end: number;
}

export const KIND_NAMES: Record<Kind, string> = { h2d: 'H2D copy', kernel: 'kernel', d2h: 'D2H copy' };

export const copySeconds = (g: GpuSpec, bytes: number) => bytes / (g.pcieGbPerS * 1e9);

/** Byte size of each chunk; the last one takes what is left. */
export function chunkBytes(p: Params): number[] {
	const total = p.mib * MIB;
	const size = Math.ceil(total / p.chunks);
	return [...Array(p.chunks).keys()].map((c) => Math.min(size, total - c * size));
}

function engineFor(kind: Kind, engines: 1 | 2): Engine {
	if (kind === 'kernel') return 'compute';
	return engines === 2 ? kind : 'copy';
}

/** Every operation with its start and end time, in issue order. */
export function schedule(p: Params): Op[] {
	const fixed = p.overheadUs * 1e-6;
	const ops: Op[] = [];
	const dur: number[] = [];
	chunkBytes(p).forEach((bytes, chunk) => {
		const copy = copySeconds(p.gpu, bytes) * (p.pinned ? 1 : 2);
		for (const kind of ['h2d', 'kernel', 'd2h'] as Kind[]) {
			ops.push({ kind, chunk, stream: chunk % p.streams, engine: engineFor(kind, p.engines), issue: ops.length, bytes, start: 0, end: 0 });
			dur.push(fixed + (kind === 'kernel' ? p.kernelRatio * copySeconds(p.gpu, bytes) : copy));
		}
	});

	if (!p.pinned) {
		let t = 0;
		ops.forEach((op, i) => {
			op.start = t;
			op.end = t += dur[i];
		});
		return ops;
	}

	const free: Record<Engine, number> = { h2d: 0, d2h: 0, copy: 0, compute: 0 };
	// Index of the next unscheduled op in each stream; ops in a stream are in issue order.
	const byStream = [...Array(p.streams).keys()].map((s) => ops.filter((o) => o.stream === s));
	const next = byStream.map(() => 0);
	const streamEnd = byStream.map(() => 0);
	for (let done = 0; done < ops.length; done++) {
		let best: Op | null = null;
		let bestStart = Infinity;
		byStream.forEach((list, s) => {
			const op = list[next[s]];
			if (!op) return;
			const start = Math.max(streamEnd[s], free[op.engine]);
			if (start < bestStart - 1e-15 || (Math.abs(start - bestStart) <= 1e-15 && op.issue < best!.issue)) {
				best = op;
				bestStart = start;
			}
		});
		const op = best!;
		op.start = bestStart;
		op.end = bestStart + dur[op.issue];
		free[op.engine] = op.end;
		streamEnd[op.stream] = op.end;
		next[op.stream] += 1;
	}
	return ops;
}

export const makespan = (ops: Op[]) => Math.max(...ops.map((o) => o.end));

/** One chunk in one stream, pinned: copy all, compute all, copy all back. */
export const serialParams = (p: Params): Params => ({ ...p, streams: 1, chunks: 1, pinned: true });

/** Totals for the comparison line and the exercises. All times are model estimates. */
export function summary(p: Params) {
	const ops = schedule(p);
	const total = makespan(ops);
	const serial = makespan(schedule(serialParams(p)));
	const busy = (e: Engine) => ops.filter((o) => o.engine === e).reduce((a, o) => a + o.end - o.start, 0);
	const engines: Engine[] = p.engines === 2 ? ['h2d', 'compute', 'd2h'] : ['copy', 'compute'];
	// No schedule can beat the busiest engine.
	const bound = Math.max(...engines.map(busy));
	return { total, serial, speedup: serial / total, bound, busy: Object.fromEntries(engines.map((e) => [e, busy(e)])) as Partial<Record<Engine, number>> };
}

/** Step 0 is the issued but unstarted pipeline; step k > 0 is the k-th distinct start or end time. */
export function eventTimes(ops: Op[]): number[] {
	const ts = [...new Set(ops.flatMap((o) => [o.start, o.end]).map((t) => +t.toPrecision(12)))].sort((a, b) => a - b);
	return ts;
}

export const lastStep = (p: Params) => eventTimes(schedule(p)).length;

export interface State {
	step: number;
	/** Seconds since the first operation started; -1 at step 0. */
	time: number;
	started: Op[];
	finished: Op[];
	running: Op[];
	done: number;
	/** Source lines to highlight, 1-based. */
	lines: number[];
}

const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));

export function stateAt(p: Params, step: number): State {
	const ops = schedule(p);
	const ts = eventTimes(ops);
	const k = Math.min(Math.max(0, Math.round(step)), ts.length);
	if (k === 0) return { step: 0, time: -1, started: [], finished: [], running: [], done: 0, lines: [line(p, 'for (int c')] };
	const t = ts[k - 1];
	const started = ops.filter((o) => near(o.start, t));
	const finished = ops.filter((o) => near(o.end, t));
	const running = ops.filter((o) => o.start <= t + 1e-15 && o.end > t + 1e-15 && !near(o.end, t));
	const done = ops.filter((o) => o.end <= t + 1e-15 || near(o.end, t)).length;
	const kinds = new Set((started.length ? started : finished).map((o) => o.kind));
	const lines = [...kinds].flatMap((kd) => KIND_LINE[kd].map((text) => line(p, text)));
	if (k === ts.length) lines.push(line(p, 'cudaDeviceSynchronize'));
	return { step: k, time: t, started, finished, running, done, lines };
}

const KIND_LINE: Record<Kind, string[]> = {
	h2d: ['cudaMemcpyAsync(d + off', 'cudaMemcpyHostToDevice'],
	kernel: ['work<<<'],
	d2h: ['cudaMemcpyAsync(h + off', 'cudaMemcpyDeviceToHost'],
};

export const CODE: Record<'pinned' | 'pageable', string> = {
	pinned: `float *h, *d;
cudaMallocHost(&h, bytes);  // pinned
cudaMalloc(&d, bytes);
cudaStream_t s[NSTREAMS];
for (int i = 0; i < NSTREAMS; i++)
    cudaStreamCreate(&s[i]);
size_t chunk = (n + NCHUNKS - 1) / NCHUNKS;
for (int c = 0; c < NCHUNKS; c++) {
    cudaStream_t st = s[c % NSTREAMS];
    size_t off = c * chunk, m = min(chunk, n - off);
    cudaMemcpyAsync(d + off, h + off, m * 4,
                    cudaMemcpyHostToDevice, st);
    work<<<(m + 255) / 256, 256, 0, st>>>(d + off, m);
    cudaMemcpyAsync(h + off, d + off, m * 4,
                    cudaMemcpyDeviceToHost, st);
}
cudaDeviceSynchronize();`,
	pageable: '',
};
CODE.pageable = CODE.pinned.replace('cudaMallocHost(&h, bytes);  // pinned', 'h = (float *)malloc(bytes);  // pageable');

function line(p: Params, text: string) {
	const i = CODE[p.pinned ? 'pinned' : 'pageable'].split('\n').findIndex((l) => l.includes(text));
	if (i < 0) throw new Error(`no line with ${text}`);
	return i + 1;
}

export const fmt = (n: number) => n.toLocaleString('en-US');
export const fmtMiB = (b: number) => `${+(b / MIB).toFixed(2)} MiB`;

export function fmtTime(s: number) {
	if (s === 0) return '0';
	const [scale, unit] = s >= 1 ? [1, 's'] : s >= 1e-3 ? [1e3, 'ms'] : s >= 1e-6 ? [1e6, 'µs'] : [1e9, 'ns'];
	return `${(s * scale).toPrecision(3)} ${unit}`;
}

export const fmtX = (x: number) => `${x.toFixed(2)}x`;

export const engineName = (e: Engine) => ({ h2d: 'H2D copy engine', d2h: 'D2H copy engine', copy: 'copy engine', compute: 'compute engine' })[e];

const opName = (o: Op) => `${KIND_NAMES[o.kind]} ${o.chunk} (stream ${o.stream})`;
const list = (xs: string[]) => (xs.length <= 2 ? xs.join(' and ') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);

/** Why an operation starts when it does: its stream, its engine, or both. */
function reason(ops: Op[], o: Op): string {
	if (o.start === 0) return 'nothing is ahead of it';
	const prev = ops.filter((x) => x.stream === o.stream && x.issue < o.issue).at(-1);
	const streamWait = prev !== undefined && near(prev.end, o.start);
	const engineWait = ops.some((x) => x !== prev && x.engine === o.engine && near(x.end, o.start));
	if (streamWait && engineWait) return `${KIND_NAMES[prev!.kind]} ${prev!.chunk} in its stream is done and the ${engineName(o.engine)} is free`;
	if (streamWait) return `${KIND_NAMES[prev!.kind]} ${prev!.chunk} before it in stream ${o.stream} is done`;
	return `the ${engineName(o.engine)} became free`;
}

export function describe(p: Params, step: number): string {
	const s = stateAt(p, step);
	const sm = summary(p);
	if (s.step === 0) {
		const where = p.chunks === 1 ? 'one chunk' : `${p.chunks} chunks of ${fmtMiB(chunkBytes(p)[0])}`;
		const intro = `The host has issued ${p.chunks * 3} operations, ${where}, each an H2D copy, a kernel and a D2H copy, into ${p.streams === 1 ? 'one stream' : `${p.streams} streams`}. Nothing has run yet.`;
		if (!p.pinned) return `${intro} The host buffer is pageable, so each cudaMemcpyAsync behaves synchronously and nothing overlaps. Predict the total before you step.`;
		return `${intro} Predict: which operations can run at the same time with ${p.engines === 2 ? 'two copy engines' : 'one copy engine'}?`;
	}
	const ops = schedule(p);
	const parts: string[] = [];
	if (s.finished.length) parts.push(`${list(s.finished.map(opName))} finished`);
	for (const o of s.started) parts.push(`${opName(o)} starts on the ${engineName(o.engine)}, because ${p.pinned ? reason(ops, o) : 'pageable memory makes every operation wait for the one before it'}`);
	const at = `t = ${fmtTime(s.time)}: ${parts.join('. ')}.`;
	if (s.step === lastStep(p))
		return `${at} All ${fmt(s.done)} operations are done in ${fmtTime(sm.total)}, against ${fmtTime(sm.serial)} for one chunk in one stream: ${fmtX(sm.speedup)} (model estimate). The busiest engine alone needs ${fmtTime(sm.bound)}.`;
	const together = s.running.length > 1 ? ` ${s.running.length} operations are running at once.` : '';
	return `${at}${together} ${fmt(s.done)} of ${fmt(p.chunks * 3)} done.`;
}
