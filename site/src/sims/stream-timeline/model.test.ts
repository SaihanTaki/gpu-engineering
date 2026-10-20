import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import {
	CHUNK_COUNTS,
	CODE,
	DEFAULTS,
	describe,
	eventTimes,
	KERNEL_RATIOS,
	lastStep,
	makespan,
	MIB,
	OVERHEAD_US,
	schedule,
	SIZES_MIB,
	stateAt,
	STREAM_COUNTS,
	summary,
	type Params,
} from './model';

const a100 = gpuById('a100');
const p = (extra: Partial<Params> = {}): Params => ({ gpu: a100, ...DEFAULTS, ...extra });
// 64 MiB over 32 GB/s; one of 8 chunks; one operation on one chunk with the 5 µs fixed cost.
const T = (64 * MIB) / 32e9;
const c = T / 8;
const d = c + 5e-6;

suite('A100, 64 MiB, kernel time = copy time, 5 µs per operation', () => {
	it('serial: copy all, compute all, copy all back', () => {
		expect(summary(p()).serial).toBeCloseTo(3 * T + 15e-6, 12);
	});
	it('one stream gives no overlap: 24 operations back to back', () => {
		expect(makespan(schedule(p({ streams: 1 })))).toBeCloseTo(24 * d, 12);
	});
	it('four streams, two copy engines: a three-stage pipeline of 8 chunks takes 10 steps', () => {
		const s = summary(p({ streams: 4 }));
		expect(s.total).toBeCloseTo(10 * d, 12);
		expect(s.bound).toBeCloseTo(8 * d, 12);
	});
	it('two streams: each is a chain of 12 operations, the second one step behind', () => {
		expect(summary(p({ streams: 2 })).total).toBeCloseTo(13 * d, 12);
	});
	it('one copy engine carries both directions, so copies alone take 16 steps', () => {
		const s = summary(p({ streams: 4, engines: 1 }));
		expect(s.bound).toBeCloseTo(16 * d, 12);
		expect(s.total).toBeGreaterThanOrEqual(16 * d - 1e-12);
	});
	it('pageable: fully serial, with the staging copy doubling each transfer', () => {
		const s = summary(p({ streams: 4, pinned: false }));
		expect(s.total).toBeCloseTo(8 * (2 * c + 2 * c + c + 3 * 5e-6), 12);
		expect(s.speedup).toBeLessThan(1);
	});
	it('more chunks shrink pipeline fill but add fixed costs', () => {
		const q = (chunks: number, overheadUs: number) => summary(p({ streams: 4, chunks, overheadUs })).total;
		expect(q(16, 0)).toBeLessThan(q(8, 0));
		expect(q(16, 20)).toBeGreaterThan(q(16, 0));
	});
	it('a ragged last chunk keeps every byte', () => {
		const ops = schedule(p({ mib: 16, chunks: 16 }));
		expect(ops.filter((o) => o.kind === 'h2d').reduce((a, o) => a + o.bytes, 0)).toBe(16 * MIB);
	});
});

suite('every combination', () => {
	const all: Params[] = [];
	for (const g of GPUS)
		for (const streams of STREAM_COUNTS)
			for (const chunks of CHUNK_COUNTS)
				for (const engines of [1, 2] as const)
					for (const pinned of [true, false])
						for (const kernelRatio of [KERNEL_RATIOS[0], 2])
							all.push({ gpu: g, streams, chunks, mib: SIZES_MIB[0], kernelRatio, pinned, engines, overheadUs: OVERHEAD_US[1] });

	it.each(all.map((q) => [q.gpu.id, q.streams, q.chunks, q.engines, q.pinned, q.kernelRatio, q] as const))(
		'%s %i streams %i chunks %i engines pinned=%s k=%f',
		(_g, _s, _c, _e, _p, _k, q) => {
			const ops = schedule(q);
			expect(ops.length).toBe(3 * q.chunks);
			const sm = summary(q);
			expect(sm.total).toBeGreaterThanOrEqual(sm.bound - 1e-12);
			const work = ops.reduce((a, o) => a + o.end - o.start, 0);
			expect(sm.total).toBeLessThanOrEqual(work + 1e-12);
			// Stream order: each op starts after the previous one in its stream ends.
			for (let s = 0; s < q.streams; s++) {
				const list = ops.filter((o) => o.stream === s);
				for (let i = 1; i < list.length; i++) expect(list[i].start).toBeGreaterThanOrEqual(list[i - 1].end - 1e-15);
			}
			// An engine runs one op at a time.
			for (const e of ['h2d', 'd2h', 'copy', 'compute']) {
				const list = ops.filter((o) => o.engine === e).sort((a, b) => a.start - b.start);
				for (let i = 1; i < list.length; i++) expect(list[i].start).toBeGreaterThanOrEqual(list[i - 1].end - 1e-15);
			}
			if (!q.pinned) for (let i = 1; i < ops.length; i++) expect(ops[i].start).toBeCloseTo(ops[i - 1].end, 15);

			const last = lastStep(q);
			expect(last).toBe(eventTimes(ops).length);
			const lines = CODE.pinned.split('\n').length;
			let prevDone = 0;
			for (let k = 0; k <= last; k++) {
				const st = stateAt(q, k);
				expect(st.done).toBeGreaterThanOrEqual(prevDone);
				prevDone = st.done;
				expect(st.lines.length).toBeGreaterThan(0);
				for (const l of st.lines) expect(l).toBeLessThanOrEqual(lines);
				if (k > 0) expect(st.started.length + st.finished.length).toBeGreaterThan(0);
				expect(describe(q, k).length).toBeGreaterThan(40);
			}
			expect(stateAt(q, last).done).toBe(ops.length);
		},
	);
});
