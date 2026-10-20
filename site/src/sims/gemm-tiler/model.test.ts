import { describe as suite, expect, it } from 'vitest';
import { gpuById } from '@sim-core/gpu-specs';
import {
	analyze,
	analyzeStep,
	autotune,
	describe,
	effectiveTile,
	LAST_STEP,
	passesOf,
	presetTiles,
	resources,
	STEP_IDS,
	warpSharedLoads,
	type Config,
	type Tile,
} from './model';

const a100 = gpuById('a100');
const t4 = gpuById('t4');
const cfg = (over: Partial<Config> = {}): Config => ({ gpu: 'a100', size: 4096, tiles: presetTiles(), ...over });
const tile = (bm: number, bn: number, bk: number, tm = 1, tn = 1): Tile => ({ bm, bn, bk, tm, tn });

suite('global traffic per output', () => {
	it('naive: 32 sectors of A and 1 of B per warp and k, so 33 bytes per output per k', () => {
		const a = analyzeStep(cfg(), 'naive');
		// 33 * 4096 + a 32-byte sector for the uncoalesced store of C.
		expect(a.dramBytes).toBe(33 * 4096 + 32);
		expect(a.globalFloats).toBe(2 * 4096);
		expect(a.intensity).toBeCloseTo(8192 / 135200, 6);
	});

	it('coalesced: 1 sector of A and 4 of B per warp and k, 5 bytes per output per k', () => {
		expect(analyzeStep(cfg(), 'coalesced').dramBytes).toBe(5 * 4096 + 4);
	});

	it('tiled: intensity BM * BN / (2 (BM + BN)) for whole-sector tiles', () => {
		// 32 x 32 tiles: K (1/32 + 1/32) floats = 256 floats = 1024 bytes, plus the 4-byte store.
		const a = analyzeStep(cfg(), 'smem');
		expect(a.globalFloats).toBe(256);
		expect(a.dramBytes).toBe(1028);
		// 128 x 128: 4096 * 2 / 128 * 4 = 256 bytes + 4.
		expect(analyzeStep(cfg(), 'reg2d').dramBytes).toBe(260);
	});

	it('BK = 4 rows of A are 16 bytes but cost a whole 32-byte sector', () => {
		const a = analyze(a100, 4096, 'smem', tile(32, 32, 4));
		// Per k-tile: 32 rows of A, 1 sector each, and 4 rows of B, 4 sectors each: (32 + 16) * 32 bytes per 1024 outputs.
		expect(a.dramBytes).toBeCloseTo((4096 / 4) * ((48 * 32) / 1024) + 4, 9);
	});

	it('reuse: each A element is loaded N / BN times', () => {
		expect(analyzeStep(cfg(), 'naive').reuse).toEqual({ a: 4096, b: 4096 });
		expect(analyzeStep(cfg(), 'reg1d').reuse).toEqual({ a: 64, b: 64 });
	});
});

suite('shared memory passes', () => {
	it('a broadcast is one pass, 32 words in 32 banks is one pass', () => {
		expect(passesOf(Array.from({ length: 32 }, () => [7]))).toBe(1);
		expect(passesOf(Array.from({ length: 32 }, (_, l) => [l]))).toBe(1);
	});

	it('16-byte loads of 128 distinct words need at least 4 passes', () => {
		expect(passesOf(Array.from({ length: 32 }, (_, l) => [4 * l, 4 * l + 1, 4 * l + 2, 4 * l + 3]))).toBe(4);
	});

	it('2D blocking at 128 x 128, 8 x 8: A loads are 2-way and B loads 4-way conflicts', () => {
		const loads = warpSharedLoads('reg2d', tile(128, 128, 8, 8, 8));
		// The warp covers 2 thread rows 64 words apart (same bank) and 16 thread columns 8 words apart.
		expect(loads.filter((l) => l.what === 'A').map((l) => passesOf(l.words))).toEqual(new Array(8).fill(2));
		expect(loads.filter((l) => l.what === 'B').map((l) => passesOf(l.words))).toEqual(new Array(8).fill(4));
	});

	it('vectorized: 2 A passes and 8 B passes per k at 128 x 128, 8 x 8', () => {
		const loads = warpSharedLoads('vec', tile(128, 128, 8, 8, 8));
		expect(loads.map((l) => passesOf(l.words))).toEqual([1, 1, 4, 4]);
	});

	it('passes per output over K = 4096', () => {
		// smem: (1 + 1) passes per warp per k for 32 outputs, plus (32 + 32) / 32 store passes per 1024 outputs.
		expect(analyzeStep(cfg(), 'smem').sharedPasses).toBeCloseTo(4096 * (2 / 32 + 64 / 32 / 1024), 9);
		// reg2d: 48 passes per 2048 outputs, plus 256 / 32 per 16384.
		expect(analyzeStep(cfg(), 'reg2d').sharedPasses).toBeCloseTo(4096 * (48 / 2048 + 8 / 16384), 9);
		expect(analyzeStep(cfg(), 'vec').sharedPasses).toBeCloseTo(4096 * (10 / 2048 + 8 / 16384), 9);
	});
});

suite('estimate', () => {
	it('A100 presets climb step by step and the vectorized kernel reaches the compute roof', () => {
		const est = STEP_IDS.map((s) => analyzeStep(cfg(), s).gflops!);
		for (let i = 1; i < est.length - 1; i++) expect(est[i]).toBeGreaterThan(est[i - 1]);
		expect(est[5]).toBe(19500);
		expect(analyzeStep(cfg(), 'naive').limit).toBe('dram');
		expect(analyzeStep(cfg(), 'smem').limit).toBe('shared');
	});

	it('A100 naive: 8192 / 135200 FLOP/B x 2039 GB/s', () => {
		expect(analyzeStep(cfg(), 'naive').gflops).toBeCloseTo((8192 / 135200) * 2039, 6);
	});

	it('A100 smem: shared roof 8192 x 108 SMs x 1.4106 GHz / 264.0 passes', () => {
		const a = analyzeStep(cfg(), 'smem');
		expect(a.sharedPasses).toBeCloseTo(264, 9);
		expect(a.gflops).toBeCloseTo((8192 * 108 * (19.5e3 / (108 * 64 * 2))) / 264, 6);
	});

	it('T4: 1D blocking is shared-limited, its DRAM roof is 15.9 FLOP/B x 300 GB/s', () => {
		const a = analyzeStep(cfg({ gpu: 't4' }), 'reg1d');
		expect(a.roofs.dram).toBeCloseTo((8192 / 516) * 300, 6);
		expect(a.limit).toBe('shared');
	});
});

suite('resources and limits', () => {
	it('counts threads, shared bytes and a register estimate', () => {
		const r = resources(a100, 'reg2d', tile(128, 128, 8, 8, 8));
		expect(r.threads).toBe(256);
		expect(r.sharedBytes).toBe(4 * (128 * 8 + 8 * 128));
		expect(r.regsPerThread).toBe(24 + 64 + 16);
		expect(r.errors).toEqual([]);
	});

	it('too many threads: 64 x 64 one output each is 4096 threads', () => {
		const a = analyze(a100, 4096, 'smem', tile(64, 64, 8));
		expect(a.res.errors[0]).toContain('4,096 threads');
		expect(a.gflops).toBeNull();
	});

	it('shared memory over 48 KB static, and over the per-block limit', () => {
		// 4 * (128 * 64 + 64 * 128) = 64 KB: fits the A100's 163 KB but not static shared memory.
		expect(resources(a100, 'reg2d', tile(128, 128, 64, 8, 8)).errors[0]).toContain('48 KB');
		// 4 * (256 * 64 * 2) = 128 KB: more than the T4's 64 KB.
		expect(resources(t4, 'reg2d', tile(256, 256, 64, 16, 16)).errors.some((e) => e.includes('64 KB'))).toBe(true);
	});

	it('16 x 16 per thread needs about 312 registers: spills', () => {
		expect(resources(a100, 'reg2d', tile(256, 256, 8, 16, 16)).errors.some((e) => e.includes('spills'))).toBe(true);
	});

	it('float4 needs multiples of 4', () => {
		expect(resources(a100, 'vec', tile(64, 64, 8, 2, 8)).errors.some((e) => e.includes('float4'))).toBe(true);
	});

	it('blocks per SM follow the tightest limit', () => {
		// 256 threads x 104 regs: 8 warps x 3328 = 26,624 registers, so 2 blocks fit in 65,536.
		const r = resources(a100, 'reg2d', tile(128, 128, 8, 8, 8));
		expect(r.blocksPerSm).toBe(2);
		expect(r.blockLimit).toBe('registers');
		// 1024 threads on a T4 (32 warps per SM): 1 block.
		expect(resources(t4, 'naive', tile(32, 32, 1)).blocksPerSm).toBe(1);
	});
});

suite('steps', () => {
	it('fixes the fields a step does not use', () => {
		const c = cfg();
		c.tiles.smem = tile(16, 16, 16, 4, 4);
		expect(effectiveTile(c, 'smem')).toEqual(tile(16, 16, 16, 1, 1));
		c.tiles.reg1d = tile(64, 64, 8, 8, 4);
		expect(effectiveTile(c, 'reg1d').tn).toBe(1);
	});

	it('autotune searches 432 configs and keeps one that runs', () => {
		const r = autotune(a100, 4096);
		expect(r.searched).toBe(4 * 4 * 3 * 3 * 3);
		expect(r.valid).toBeGreaterThan(0);
		expect(r.best!.gflops).toBe(19500);
		expect(r.tied).toBeGreaterThan(1);
		expect(analyzeStep(cfg(), 'autotune').res.errors).toEqual([]);
	});

	it('every step has a status line', () => {
		for (let s = 0; s <= LAST_STEP; s++) expect(describe(cfg(), s)).toMatch(/Model estimate|cannot run/);
		const c = cfg();
		c.tiles.smem = tile(64, 64, 8);
		expect(describe(c, 2)).toContain('cannot run');
	});
});
