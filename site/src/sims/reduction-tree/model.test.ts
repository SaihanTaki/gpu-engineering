import { describe as suite, expect, it } from 'vitest';
import {
	BLOCK_SIZES,
	CODE,
	describe,
	efficiency,
	expectedSum,
	lastStep,
	MODE_IDS,
	stepAt,
	steps,
	totalsAt,
	warpLanes,
	type Config,
} from './model';

const all = MODE_IDS.flatMap((mode) => BLOCK_SIZES.map((block) => ({ mode, block }) as Config));
const total = (c: Config) => totalsAt(c, lastStep(c));

suite('reduction tree', () => {
	it.each(all)('the block sum is right and takes blockDim.x - 1 useful adds: %o', (c) => {
		expect(stepAt(c, lastStep(c)).out).toBe(expectedSum(c));
		expect(total(c).adds).toBe(c.block - 1);
	});

	it.each(all)('exactly one value is live at the end: %o', (c) => {
		const s = steps(c).at(-2)!;
		const live = [...s.sharedLive, ...(s.regsLive ?? [])].filter(Boolean).length;
		expect(live).toBe(1);
	});

	it('64 threads, interleaved: 6 levels, both warps busy until stride 32', () => {
		const c: Config = { mode: 'interleaved', block: 64 };
		expect(steps(c).map((s) => s.stride)).toEqual([0, 1, 2, 4, 8, 16, 32, 0]);
		expect(warpLanes(stepAt(c, 1))).toEqual([16, 16]);
		expect(warpLanes(stepAt(c, 5))).toEqual([1, 1]);
		expect(warpLanes(stepAt(c, 6))).toEqual([1, 0]);
		expect(total(c)).toMatchObject({ warpIssues: 11, divergentIssues: 11, barriers: 6 });
	});

	it('64 threads, sequential: warp 1 idles from the first level', () => {
		const c: Config = { mode: 'sequential', block: 64 };
		expect(warpLanes(stepAt(c, 1))).toEqual([32, 0]);
		expect(warpLanes(stepAt(c, 2))).toEqual([16, 0]);
		expect(total(c)).toMatchObject({ warpIssues: 6, divergentIssues: 5, barriers: 6 });
		expect(efficiency(total(c))).toBeCloseTo(63 / 192, 10);
	});

	it('256 threads: 47 warp adds interleaved, 12 sequential, 12 with no divergence using shuffles', () => {
		expect(total({ mode: 'interleaved', block: 256 })).toMatchObject({ warpIssues: 47, divergentIssues: 47, barriers: 8, sharedRequests: 141 });
		expect(total({ mode: 'sequential', block: 256 })).toMatchObject({ warpIssues: 12, divergentIssues: 5, barriers: 8, sharedRequests: 36 });
		expect(total({ mode: 'warp-final', block: 256 })).toMatchObject({ warpIssues: 12, divergentIssues: 0, barriers: 2, sharedRequests: 20 });
	});

	it('shuffles follow __shfl_down_sync: lanes past 31 keep their own value', () => {
		const c: Config = { mode: 'warp-final', block: 64 };
		const regs = stepAt(c, 1).regs!;
		const after = stepAt(c, 2).regs!;
		expect(stepAt(c, 2).kind).toBe('shuffle');
		expect(after[0]).toBe(regs[0] + regs[16]);
		expect(after[20]).toBe(regs[20] * 2);
		expect(stepAt(c, 2).regsLive!.filter(Boolean).length).toBe(16);
	});

	it.each(all)('every step has a status and code lines that exist: %o', (c) => {
		const code = CODE[c.mode].split('\n');
		for (let s = 0; s <= lastStep(c); s++) {
			expect(describe(c, s).length).toBeGreaterThan(60);
			for (const line of stepAt(c, s).lines) expect(code[line - 1].trim().length).toBeGreaterThan(2);
		}
		expect(describe(c, lastStep(c))).toContain(`writes the block's sum, ${expectedSum(c)}`);
	});

	it('the add line is the one the shared levels highlight', () => {
		for (const mode of MODE_IDS) expect(CODE[mode].split('\n')[4].trim()).toBe('sdata[tid] += sdata[tid + s];');
		expect(CODE['warp-final'].split('\n')[11].trim()).toBe('v += __shfl_down_sync(FULL_MASK, v, offset);');
	});

	it('status text names the divergence', () => {
		expect(describe({ mode: 'interleaved', block: 64 }, 1)).toContain('2 warps of 2 issue the add, each with only 16 of 32 lanes on (all divergent)');
		expect(describe({ mode: 'sequential', block: 64 }, 1)).toContain('1 warp of 2 issues the add, with all 32 lanes on. 1 warp has no work and skips it');
	});
});
