import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import { describe, lastStep, occupancy, overflowAt, placedAt, usage } from './model';

const a100 = gpuById('a100');
const t4 = gpuById('t4');
const h100 = gpuById('h100');
const KB = 1024;

// Each case worked out by hand from the rules in cuda_occupancy.h; the comments show the arithmetic.
suite('occupancy, hand-computed cases', () => {
	it('A100, 32 regs, 256 threads: warps and registers both allow 8 blocks', () => {
		// 64 / 8 warps = 8. 32 * 32 = 1024 regs per warp, 16384 / 1024 = 16 warps per partition, 64 / 8 = 8.
		// Shared: 0 + 1024 reserved = 1024, 167936 / 1024 = 164.
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 0, blockSize: 256 });
		expect(o.limits).toEqual({ warps: 8, registers: 8, shared: 164, blocks: 32 });
		expect(o.blocks).toBe(8);
		expect(o.occupancy).toBe(1);
		expect(o.limiting).toEqual(['warps', 'registers']);
	});

	it('A100, 33 regs: one more register costs a quarter of the warps', () => {
		// 33 * 32 = 1056, rounded up to 1280. 16384 / 1280 = 12 warps per partition, 48 per SM, 48 / 8 = 6.
		const o = occupancy(a100, { regsPerThread: 33, smemPerBlock: 0, blockSize: 256 });
		expect(o.regsPerWarp).toBe(1280);
		expect(o.warpsPerPartition).toBe(12);
		expect(o.limits.registers).toBe(6);
		expect(o.blocks).toBe(6);
		expect(o.occupancy).toBe(0.75);
		expect(o.limiting).toEqual(['registers']);
	});

	it('A100, 32-thread blocks run out of block slots at 50%', () => {
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 0, blockSize: 32 });
		expect(o.limits).toEqual({ warps: 64, registers: 64, shared: 164, blocks: 32 });
		expect(o.blocks).toBe(32);
		expect(o.occupancy).toBe(0.5);
		expect(o.limiting).toEqual(['blocks']);
	});

	it('T4, 64-thread blocks fill the 32 warp slots with 16 blocks; no shared memory is reserved', () => {
		const o = occupancy(t4, { regsPerThread: 32, smemPerBlock: 0, blockSize: 64 });
		expect(o.smemPerBlock).toBe(0);
		expect(o.limits).toEqual({ warps: 16, registers: 32, shared: Infinity, blocks: 16 });
		expect(o.occupancy).toBe(1);
		expect(o.limiting).toEqual(['warps', 'blocks']);
	});

	it('A100, 48 KB per block of 128 threads: shared memory allows 3 blocks', () => {
		// 49152 + 1024 = 50176, a multiple of 128. 167936 / 50176 = 3.35, so 3. 3 * 4 warps = 12 of 64.
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 48 * KB, blockSize: 128 });
		expect(o.smemPerBlock).toBe(50176);
		expect(o.limits.shared).toBe(3);
		expect(o.occupancy).toBe(12 / 64);
		expect(o.limiting).toEqual(['shared']);
	});

	it('rounds shared memory to 256 bytes on Turing and 128 bytes after the reservation on Ampere', () => {
		// T4: 1000 -> 1024, 65536 / 1024 = 64. A100: 1000 + 1024 = 2024 -> 2048, 167936 / 2048 = 82.
		expect(occupancy(t4, { regsPerThread: 32, smemPerBlock: 1000, blockSize: 256 }).limits.shared).toBe(64);
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 1000, blockSize: 256 });
		expect(o.smemPerBlock).toBe(2048);
		expect(o.limits.shared).toBe(82);
	});

	it('H100, 100 KB and 64 regs at 256 threads: shared memory allows 2 blocks, 25%', () => {
		// Shared: 102400 + 1024 = 103424, 233472 / 103424 = 2.26. Registers: 2048 per warp, 8 per partition, 32 / 8 = 4.
		const o = occupancy(h100, { regsPerThread: 64, smemPerBlock: 100 * KB, blockSize: 256 });
		expect(o.limits).toEqual({ warps: 8, registers: 4, shared: 2, blocks: 32 });
		expect(o.occupancy).toBe(0.25);
	});

	it('1024 threads at 64 regs fit once; at 65 regs the launch fails', () => {
		// 64 regs: 2048 per warp x 32 warps = 65536, exactly the per-block limit.
		const ok = occupancy(a100, { regsPerThread: 64, smemPerBlock: 0, blockSize: 1024 });
		expect(ok.blocks).toBe(1);
		expect(ok.occupancy).toBe(0.5);
		expect(occupancy(t4, { regsPerThread: 64, smemPerBlock: 0, blockSize: 1024 }).occupancy).toBe(1);
		// 65 regs: 2080 -> 2304 per warp, x 32 = 73728 > 65536.
		const bad = occupancy(a100, { regsPerThread: 65, smemPerBlock: 0, blockSize: 1024 });
		expect(bad.blocks).toBe(0);
		expect(bad.error).toContain('73,728');
	});

	it('the register check rounds warps up to a multiple of the 4 partitions', () => {
		// 800 threads = 25 warps at 80 regs (2560 per warp): 25 x 2560 = 64000 fits the 65536 limit,
		// but 28 x 2560 = 71680 does not, so the launch fails.
		const o = occupancy(a100, { regsPerThread: 80, smemPerBlock: 0, blockSize: 800 });
		expect(o.regsPerBlock).toBe(64000);
		expect(o.blocks).toBe(0);
		expect(o.error).toContain('71,680');
		// 992 threads = 31 warps at 64 regs: rounded to 32 x 2048 = 65536, which fits. One block, 31 of 64 warps.
		const p = occupancy(a100, { regsPerThread: 64, smemPerBlock: 0, blockSize: 992 });
		expect(p.blocks).toBe(1);
		expect(p.occupancy).toBe(31 / 64);
	});

	it('the per-block shared memory limit leaves out the reserved 1 KB', () => {
		expect(occupancy(a100, { regsPerThread: 32, smemPerBlock: 163 * KB, blockSize: 256 }).blocks).toBe(1);
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 164 * KB, blockSize: 256 });
		expect(o.blocks).toBe(0);
		expect(o.error).toContain('163 KB');
		expect(occupancy(t4, { regsPerThread: 32, smemPerBlock: 64 * KB, blockSize: 256 }).blocks).toBe(1);
	});

	it("matches the Best Practices Guide's 37-register example", () => {
		// The guide's example is for compute capability 7.0, which has the A100's register file, 64 warp slots and rounding.
		// 37 * 32 = 1184 -> 1280 per warp, 12 warps per partition, 48 per SM.
		expect(occupancy(a100, { regsPerThread: 37, smemPerBlock: 0, blockSize: 128 })).toMatchObject({ blocks: 12, occupancy: 0.75 });
		// 320 threads = 10 warps: 48 / 10 = 4 blocks, 40 of 64 warps, which the guide rounds to 63%.
		expect(occupancy(a100, { regsPerThread: 37, smemPerBlock: 0, blockSize: 320 })).toMatchObject({ blocks: 4, occupancy: 40 / 64 });
	});

	it('block sizes that are not a multiple of 32 still take whole warps', () => {
		const o = occupancy(a100, { regsPerThread: 32, smemPerBlock: 0, blockSize: 100 });
		expect(o.warpsPerBlock).toBe(4);
		expect(o.limits.warps).toBe(16);
	});
});

suite('occupancy steps', () => {
	const k = { regsPerThread: 33, smemPerBlock: 0, blockSize: 256 };
	const o = occupancy(a100, k);

	it('places one block per step, then tries one too many', () => {
		expect(lastStep(o)).toBe(7);
		expect(placedAt(o, 0)).toBe(0);
		expect(placedAt(o, 6)).toBe(6);
		expect(overflowAt(o, 6)).toBe(false);
		expect(overflowAt(o, 7)).toBe(true);
	});

	it('deals warps to partitions evenly and never past their capacity', () => {
		for (const g of GPUS)
			for (const regs of [16, 33, 64, 100, 255])
				for (const blockSize of [32, 96, 256, 1024]) {
					const oc = occupancy(g, { regsPerThread: regs, smemPerBlock: 0, blockSize });
					for (const w of usage(g, oc, oc.blocks).perPartition) expect(w).toBeLessThanOrEqual(oc.warpsPerPartition);
				}
		expect(usage(a100, o, 6).perPartition).toEqual([12, 12, 12, 12]);
		expect(usage(a100, o, 1).perPartition).toEqual([2, 2, 2, 2]);
	});

	it('describes every step and names the limit at the end', () => {
		expect(describe(a100, k, 0)).toContain('Predict');
		expect(describe(a100, k, 3)).toContain('Block 2 placed');
		const end = describe(a100, k, 7);
		expect(end).toContain('Block 6 does not fit');
		expect(end).toContain('room for 0 more warps');
		expect(end).toContain('75%');
		expect(end).toContain('Limited by registers');
	});

	it('reports a failed launch with a single step', () => {
		const bad = { regsPerThread: 65, smemPerBlock: 0, blockSize: 1024 };
		expect(lastStep(occupancy(a100, bad))).toBe(0);
		expect(describe(a100, bad, 0)).toContain('too many resources');
	});

	it('the default case from chapter 1 gives 32 registers per thread at full occupancy', () => {
		for (const g of [a100, h100]) {
			const full = occupancy(g, { regsPerThread: (g.registerFileKbPerSm * 256) / (g.maxWarpsPerSm * 32), smemPerBlock: 0, blockSize: 256 });
			expect(full.occupancy).toBe(1);
		}
	});
});
