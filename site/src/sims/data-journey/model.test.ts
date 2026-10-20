import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import { arrayBytes, blocksFor, CODE, describe, fmtBytes, fmtTime, lastStep, SIZES, stateAt, times } from './model';

const a100 = gpuById('a100');
const N = 2 ** 24;

suite('pcie specs', () => {
	it('per-direction peak is 16 GB/s at Gen3 and doubles each generation', () => {
		for (const g of GPUS) expect(g.pcieGbPerS, g.id).toBe(16 * 2 ** (g.pcieGen - 3));
		expect(GPUS.map((g) => g.pcieGen)).toEqual([3, 4, 5]);
	});
});

suite('time estimates', () => {
	it('A100 at n = 2^24, by hand', () => {
		// 64 MiB = 67,108,864 bytes per array.
		const t = times(a100, N);
		expect(t.copyIn).toBeCloseTo((2 * 67108864) / 32e9, 15);
		expect(t.kernel).toBeCloseTo((3 * 67108864) / 2039e9, 15);
		expect(t.copyOut).toBeCloseTo(67108864 / 32e9, 15);
		expect(fmtTime(t.copyIn)).toBe('4.19 ms');
		expect(fmtTime(t.kernel)).toBe('98.7 µs');
		expect(fmtTime(t.copyOut)).toBe('2.10 ms');
		expect(fmtTime(t.total)).toBe('6.39 ms');
	});

	it.each(GPUS.flatMap((g) => SIZES.map((n) => [g.id, n] as const)))('%s, n = %i: the copies take longer than the kernel', (id, n) => {
		const t = times(gpuById(id), n);
		expect(t.copyIn).toBeGreaterThan(t.kernel);
		expect(t.copyOut).toBeGreaterThan(t.kernel / 2);
		expect(t.copyIn + t.copyOut).toBeGreaterThan(10 * t.kernel);
	});

	it('the three arrays fit in GPU memory at the largest size', () => {
		for (const g of GPUS) expect(3 * arrayBytes(SIZES.at(-1)!)).toBeLessThan(g.memoryGb * 1e9);
	});

	it('sizes run from about 1K to about 1G elements', () => {
		expect(SIZES[0]).toBe(1024);
		expect(SIZES.at(-1)).toBe(2 ** 30);
	});
});

suite('steps', () => {
	it('data moves CPU RAM, HBM, back, then the device buffers are freed', () => {
		const at = (s: number) => stateAt(a100, N, s).buffers;
		expect(at(0)).toEqual({ hA: 'full', hB: 'full', hC: 'empty', dA: 'none', dB: 'none', dC: 'none' });
		expect(at(1)).toMatchObject({ dA: 'empty', dB: 'empty', dC: 'empty' });
		expect(at(2)).toMatchObject({ dA: 'full', dB: 'empty' });
		expect(at(3)).toMatchObject({ dA: 'full', dB: 'full', dC: 'empty' });
		expect(at(6)).toMatchObject({ dC: 'full', hC: 'empty' });
		expect(at(7)).toMatchObject({ hC: 'full' });
		expect(at(lastStep)).toEqual({ hA: 'full', hB: 'full', hC: 'full', dA: 'none', dB: 'none', dC: 'none' });
	});

	it('elapsed time never goes backwards and ends at the total', () => {
		let prev = -1;
		for (let s = 0; s <= lastStep; s++) {
			const d = stateAt(a100, N, s).done;
			const sum = d.copyIn + d.kernel + d.copyOut;
			expect(sum).toBeGreaterThanOrEqual(prev);
			prev = sum;
		}
		expect(prev).toBeCloseTo(times(a100, N).total, 15);
	});

	it('only the copy steps use PCIe', () => {
		const used = [...Array(lastStep + 1).keys()].filter((s) => stateAt(a100, N, s).transfer);
		expect(used).toEqual([2, 3, 7]);
		expect(stateAt(a100, N, 7).transfer).toEqual({ from: 'dC', to: 'hC' });
	});

	it('highlighted lines exist in the code', () => {
		const count = CODE.split('\n').length;
		for (let s = 0; s <= lastStep; s++) for (const l of stateAt(a100, N, s).lines) expect(l).toBeLessThanOrEqual(count);
		expect(CODE.split('\n')[13]).toContain('add<<<');
	});

	it('every step has a status line, with numbers where it matters', () => {
		for (const g of GPUS) for (let s = 0; s <= lastStep; s++) expect(describe(g, N, s).length).toBeGreaterThan(40);
		expect(describe(a100, N, 2)).toContain('64 MiB takes at least 2.10 ms');
		expect(describe(a100, N, 4)).toContain('add<<<65,536, 256>>>');
		expect(describe(a100, 1024, 4)).toContain('Only 4 of the 108 SMs');
		expect(describe(a100, N, lastStep)).toContain('PCIe is 63.7x slower');
	});

	it('formats sizes and blocks', () => {
		expect(fmtBytes(arrayBytes(1024))).toBe('4 KiB');
		expect(fmtBytes(3 * arrayBytes(2 ** 30))).toBe('12 GiB');
		expect(blocksFor(1000)).toBe(4);
		expect(blocksFor(2 ** 30)).toBe(4194304);
	});
});
