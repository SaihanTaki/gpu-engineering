import { describe as suite, expect, it } from 'vitest';
import { describe, final, fmtBytes, fmtUs, kernelAt, LAST_STEP, linesAt, speedup, totalsAt, type Config } from './model';

const cfg = (over: Partial<Config> = {}): Config => ({ gpu: 'a100', log2n: 24, launchUs: 1, ...over });
const MiB = 2 ** 20;

suite('DRAM traffic', () => {
	it('three kernels move 7 arrays, the fused one 3', () => {
		// 2^24 floats is 64 MiB per array.
		const u = final(cfg(), 'unfused');
		const f = final(cfg(), 'fused');
		expect(u.readBytes).toBe(4 * 64 * MiB);
		expect(u.writeBytes).toBe(3 * 64 * MiB);
		expect(f.readBytes).toBe(2 * 64 * MiB);
		expect(f.writeBytes).toBe(64 * MiB);
		expect(u.bytes / f.bytes).toBeCloseTo(7 / 3, 12);
	});

	it('counts grow kernel by kernel', () => {
		expect([0, 1, 2, 3, 4].map((s) => totalsAt(cfg(), 'unfused', s).bytes / (64 * MiB))).toEqual([0, 2, 5, 7, 7]);
		expect([0, 1, 2, 3, 4].map((s) => totalsAt(cfg(), 'fused', s).bytes / (64 * MiB))).toEqual([0, 3, 3, 3, 3]);
		expect(totalsAt(cfg(), 'unfused', 2).launches).toBe(2);
	});

	it('maps steps to kernels and code lines', () => {
		expect([0, 1, 2, 3, 4].map((s) => kernelAt('unfused', s))).toEqual([null, 0, 1, 2, null]);
		expect([0, 1, 2, 3, 4].map((s) => kernelAt('fused', s))).toEqual([null, 0, null, null, null]);
		expect(linesAt('unfused', 2)).toEqual([7, 15]);
		expect(linesAt('fused', 1)).toEqual([4, 5, 9]);
		expect(linesAt('fused', 2)).toEqual([]);
	});
});

suite('time estimate', () => {
	it('large n on the A100: bandwidth sets the time, ratio near 7/3', () => {
		// 7 x 64 MiB / 2039 GB/s = 230.39 us, + 3 us; 3 x 64 MiB / 2039 = 98.74 us, + 1 us.
		expect(final(cfg(), 'unfused').timeUs).toBeCloseTo(233.39, 2);
		expect(final(cfg(), 'fused').timeUs).toBeCloseTo(99.74, 2);
		expect(speedup(cfg())).toBeCloseTo(2.34, 2);
	});

	it('small n: launches set the time, ratio near 3', () => {
		// 28 KiB / 2039 GB/s = 0.0141 us; 12 KiB = 0.0060 us.
		expect(speedup(cfg({ log2n: 10 }))).toBeCloseTo(2.996, 3);
	});

	it('with no launch cost the ratio is exactly the traffic ratio', () => {
		expect(speedup(cfg({ launchUs: 0, gpu: 't4' }))).toBeCloseTo(7 / 3, 12);
	});
});

suite('text', () => {
	it('formats', () => {
		expect(fmtBytes(64 * MiB)).toBe('64 MiB');
		expect(fmtBytes(4096)).toBe('4 KiB');
		expect(fmtUs(233.39)).toBe('233 µs');
		expect(fmtUs(0.5)).toBe('500 ns');
		expect(fmtUs(1500)).toBe('1.5 ms');
		expect(fmtUs(0)).toBe('0 µs');
	});

	it('every step has a status line', () => {
		for (let s = 0; s <= LAST_STEP; s++) expect(describe(cfg(), s).length).toBeGreaterThan(40);
		expect(describe(cfg(), 2)).toContain('One fused kernel: done.');
		expect(describe(cfg(), LAST_STEP)).toContain('2.34x');
	});
});
