import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import { CODE, describe, FAULT_US, fmtTime, lastStep, MODES, PAGE_BYTES, PAGE_COUNTS, run, stateAt, summary, total, type Mode, type Params } from './model';

const a100 = gpuById('a100');
const p = (mode: Mode, extra: Partial<Params> = {}): Params => ({ gpu: a100, mode, pages: 8, faultUs: 20, ...extra });
const modes = MODES.map((m) => m.id);

suite('totals on an A100, 8 pages of 2 MiB, 20 µs per fault', () => {
	// 16 MiB / 32 GB/s = 524.288 µs; one page 65.536 µs; kernel 2 * 16 MiB / 2039 GB/s = 16.456 µs.
	it('on-demand: 16 page transfers, 16 faults', () => {
		const s = summary(p('demand'));
		expect(s.gpuFaults).toBe(8);
		expect(s.cpuFaults).toBe(8);
		expect(s.transfers).toBe(16);
		expect(s.pcie).toBeCloseTo(16 * 65.536e-6, 12);
		expect(s.fault).toBeCloseTo(320e-6, 12);
		expect(fmtTime(s.total)).toBe('1.39 ms');
	});
	it('prefetch and pinned copies: two bulk transfers, no faults, the same lower bound', () => {
		for (const m of ['prefetch', 'pinned'] as const) {
			const s = summary(p(m));
			expect(s.gpuFaults + s.cpuFaults).toBe(0);
			expect(s.transfers).toBe(2);
			expect(s.pcie).toBeCloseTo(2 * 524.288e-6, 12);
			expect(fmtTime(s.total)).toBe('1.07 ms');
		}
	});
	it('pageable adds two staging copies', () => {
		const s = summary(p('pageable'));
		expect(s.staging).toBeCloseTo(2 * 524.288e-6, 12);
		expect(fmtTime(s.total)).toBe('2.11 ms');
	});
	it('with zero fault cost, on-demand equals prefetch', () => {
		expect(summary(p('demand', { faultUs: 0 })).total).toBeCloseTo(summary(p('prefetch')).total, 12);
	});
});

suite('every mode, GPU and size', () => {
	const all = GPUS.flatMap((g) => PAGE_COUNTS.flatMap((pages) => FAULT_US.flatMap((faultUs) => modes.map((mode) => ({ gpu: g, mode, pages, faultUs }) as Params))));

	it.each(all.map((q) => [q.gpu.id, q.mode, q.pages, q.faultUs, q] as const))('%s %s %i pages %i µs', (_g, _m, _n, _f, q) => {
		const states = run(q);
		expect(states.length - 1).toBe(lastStep(q));
		// Every byte crosses PCIe once each way.
		const end = states.at(-1)!;
		expect(end.bytes).toBe(2 * q.pages * PAGE_BYTES);
		let prev = -1;
		const lines = CODE[q.mode].split('\n').length;
		for (let i = 0; i < states.length; i++) {
			const s = states[i];
			expect(total(s.done)).toBeGreaterThanOrEqual(prev);
			prev = total(s.done);
			for (const l of s.lines) expect(l).toBeLessThanOrEqual(lines);
			expect(s.lines.length).toBeGreaterThan(0);
			expect(describe(q, i).length).toBeGreaterThan(40);
			if (q.mode === 'demand' || q.mode === 'prefetch') {
				// A managed page lives in one place at a time.
				for (let k = 0; k < q.pages; k++) expect(s.cpu[k] === 'none' || s.gpu[k] === 'none').toBe(true);
			}
		}
		expect(end.cpu.every((c) => c === 'none') && end.gpu.every((c) => c === 'none')).toBe(true);
	});

	it('faulting costs more than prefetching whenever a fault costs anything', () => {
		for (const g of GPUS) expect(summary({ gpu: g, mode: 'demand', pages: 16, faultUs: 10 }).total).toBeGreaterThan(summary({ gpu: g, mode: 'prefetch', pages: 16, faultUs: 10 }).total);
	});
});

suite('steps', () => {
	it('on-demand moves one page per GPU fault, in order', () => {
		const states = run(p('demand'));
		const faults = states.filter((s) => s.id === 'gpuFault');
		expect(faults.map((s) => s.fault!.page)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
		expect(faults[2].gpu.slice(0, 4)).toEqual(['2x', '2x', '2x', 'none']);
		expect(faults[2].cpu.slice(0, 4)).toEqual(['none', 'none', 'none', 'x']);
		expect(states.length).toBe(2 * 8 + 5);
	});

	it('the kernel line is highlighted while the GPU faults', () => {
		const s = stateAt(p('demand'), 3);
		expect(s.id).toBe('gpuFault');
		expect(CODE.demand.split('\n')[s.lines[0] - 1]).toContain('if (i < n)');
	});

	it('pageable copies go through the staging buffer', () => {
		const ids = run(p('pageable')).map((s) => s.id);
		expect(ids).toEqual(['alloc', 'init', 'stageIn', 'copyIn', 'kernel', 'copyOut', 'stageOut', 'read', 'free']);
		expect(stateAt(p('pageable'), 2).staging).toBe('x');
		expect(stateAt(p('pageable'), 5).cpu[0]).toBe('x');
		expect(stateAt(p('pageable'), 6).cpu[0]).toBe('2x');
	});

	it('prefetch code uses the CUDA 13 cudaMemLocation form', () => {
		expect(CODE.prefetch).toContain('cudaMemPrefetchAsync(x, bytes, gpu, 0, stream)');
		expect(CODE.prefetch).toContain('cudaMemLocationTypeHost');
	});

	it('status text carries the numbers', () => {
		expect(describe(p('demand'), 3)).toContain('at least 65.5 µs');
		expect(describe(p('prefetch'), 2)).toContain('at least 524 µs');
		expect(describe(p('demand'), lastStep(p('demand')))).toContain('8 GPU faults, 8 CPU faults, 16 transfers');
	});
});
