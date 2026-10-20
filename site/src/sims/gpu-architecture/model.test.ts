import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import { components, describe, formatKb, LEVELS, lastStep, memoryLevels, totals } from './model';

suite('gpu specs', () => {
	// Totals as NVIDIA states them for each product, to catch a wrong per-SM number.
	it.each([
		['t4', 2560, 320],
		['a100', 6912, 432],
		['h100', 16896, 528],
	])('%s per-SM counts multiply out to the published totals', (id, cores, tensor) => {
		const t = totals(gpuById(id));
		expect(t.cudaCores).toBe(cores);
		expect(t.tensorCores).toBe(tensor);
	});

	// FP64 is left out: Turing has 2 FP64 units per SM, outside the partitions.
	it('per-SM units divide evenly into partitions', () => {
		for (const g of GPUS) {
			for (const n of [g.fp32PerSm, g.int32PerSm, g.tensorPerSm, g.sfuPerSm, g.registerFileKbPerSm]) {
				expect(Number.isInteger(n / g.partitionsPerSm), `${g.id}: ${n}`).toBe(true);
			}
		}
	});
});

suite('gpu architecture model', () => {
	it('has one step per zoom level', () => {
		expect(lastStep).toBe(LEVELS.length - 1);
	});

	it('memory levels grow in capacity from registers to DRAM', () => {
		for (const g of GPUS) {
			const sizes = memoryLevels(g).map((m) => m.totalKb);
			expect(sizes[3]).toBeGreaterThan(sizes[2]);
			expect(sizes.slice(0, 3).every((kb) => kb < sizes[3])).toBe(true);
		}
	});

	it('T4 has more register storage than L2, A100 and H100 do not', () => {
		expect(describe(gpuById('t4'), 3)).toContain('hold more than');
		expect(describe(gpuById('a100'), 3)).not.toContain('hold more than');
		expect(describe(gpuById('h100'), 3)).not.toContain('hold more than');
	});

	it('every level lists components with text', () => {
		for (const g of GPUS)
			for (const level of LEVELS) {
				const cs = components(g, level);
				expect(cs.length).toBeGreaterThan(0);
				for (const c of cs) expect(c.detail.length).toBeGreaterThan(10);
			}
	});

	it('formats sizes in binary units', () => {
		expect(formatKb(256)).toBe('256 KB');
		expect(formatKb(108 * 256)).toBe('27 MB');
		expect(formatKb(80 * 1024 * 1024)).toBe('80 GB');
	});
});
