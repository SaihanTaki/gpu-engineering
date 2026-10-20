import { describe as suite, expect, it } from 'vitest';
import { GPUS, gpuById } from '@sim-core/gpu-specs';
import { PEAK_FP32_TFLOPS } from '../transpose/model';
import {
	attainable,
	describe,
	evaluate,
	kernelPoint,
	LAST_STEP,
	peakTflops,
	PRESETS,
	presetAi,
	presetAt,
	ridge,
	roofFor,
	roofsOf,
	type Config,
} from './model';

const t4 = gpuById('t4');
const a100 = gpuById('a100');
const h100 = gpuById('h100');
const cfg = (over: Partial<Config> = {}): Config => ({ gpu: 'a100', roof: 'fp32', log2Flops: 30, log2Bytes: 32, ...over });

suite('gpu-specs peak rates', () => {
	it('match the datasheet figures (dense, no sparsity)', () => {
		expect([t4, a100, h100].map((g) => g.peakFp32Tflops)).toEqual([8.1, 19.5, 67]);
		expect([t4, a100, h100].map((g) => g.peakFp16TensorTflops)).toEqual([65, 312, 989.5]);
		expect([t4, a100, h100].map((g) => g.peakInt8TensorTops)).toEqual([130, 624, 1979]);
		expect([t4, a100, h100].map((g) => g.peakTf32TensorTflops)).toEqual([null, 156, 494.5]);
		expect([t4, a100, h100].map((g) => g.peakFp8TensorTflops)).toEqual([null, null, 1979]);
	});

	it('agree with the FP32 table chapter 11 uses', () => {
		for (const g of GPUS) expect(g.peakFp32Tflops).toBe(PEAK_FP32_TFLOPS[g.id]);
	});

	it('order the roofs: FP64 <= FP32 < tensor rates', () => {
		for (const g of GPUS) {
			expect(g.peakFp64Tflops).toBeLessThanOrEqual(g.peakFp32Tflops);
			expect(g.peakFp16TensorTflops).toBeGreaterThan(g.peakFp32Tflops);
			if (g.peakTf32TensorTflops !== null) expect(g.peakTf32TensorTflops).toBeLessThan(g.peakFp16TensorTflops);
		}
	});

	it('T4 FP64 is about 1/32 of FP32, as the Turing whitepaper states', () => {
		expect(t4.peakFp64Tflops / t4.peakFp32Tflops).toBeCloseTo(1 / 32, 2);
	});
});

suite('roofline', () => {
	it('ridge point is peak over bandwidth', () => {
		// 19.5 TFLOPS / 2.039 TB/s = 9.56 FLOP/byte; 8.1 / 0.3 = 27.
		expect(ridge(a100, 'fp32')).toBeCloseTo(9.564, 3);
		expect(ridge(t4, 'fp32')).toBeCloseTo(27, 9);
		expect(ridge(h100, 'fp8')).toBeCloseTo(1979 / 3.35, 9);
	});

	it('attainable is min(peak, AI x BW)', () => {
		expect(attainable(a100, 'fp32', 1)).toBeCloseTo(2.039, 9);
		expect(attainable(a100, 'fp32', 1000)).toBe(19.5);
		expect(attainable(a100, 'fp32', ridge(a100, 'fp32'))).toBeCloseTo(19.5, 9);
	});

	it('classifies and times a kernel', () => {
		const p = kernelPoint(cfg());
		expect(p.ai).toBe(0.25);
		expect(p.bound).toBe('memory');
		// 2^32 bytes / 2.039e6 bytes per us.
		expect(p.timeUs).toBeCloseTo(2 ** 32 / 2.039e6, 6);
		const q = kernelPoint(cfg({ log2Flops: 40, log2Bytes: 30 }));
		expect(q.bound).toBe('compute');
		expect(q.fractionOfPeak).toBe(1);
		expect(q.timeUs).toBeCloseTo(2 ** 40 / 19.5e6, 6);
	});

	it('unsupported roofs are null and fall back to FP32', () => {
		expect(peakTflops(t4, 'tf32')).toBeNull();
		expect(roofsOf(t4)).toEqual(['fp64', 'fp32', 'fp16', 'int8']);
		expect(roofsOf(h100)).toHaveLength(6);
		expect(roofFor(t4, 'fp8')).toBe('fp32');
		expect(roofFor(h100, 'fp8')).toBe('fp8');
	});
});

suite('presets', () => {
	it('have the intensities the chapter quotes', () => {
		const ai = Object.fromEntries(PRESETS.map((p) => [p.id, presetAi(p)]));
		expect(ai.vadd).toBeCloseTo(1 / 12, 12);
		expect(ai.unfused).toBeCloseTo(3 / 28, 12);
		expect(ai.fused).toBe(0.25);
		expect(ai.reduce).toBe(0.25);
		expect(ai.transpose).toBe(0);
		expect(ai['gemm-naive']).toBe(0.25);
		expect(ai['gemm-tiled']).toBe(8);
		expect(ai['gemm-ideal']).toBeCloseTo(4096 / 6, 9);
	});

	it('GEMM is the only one that can be compute-bound on the A100 FP32 roof', () => {
		const bound = PRESETS.filter((p) => p.flops > 0).map((p) => [p.id, evaluate(a100, 'fp32', p.flops, p.bytes).bound]);
		expect(bound.filter(([, b]) => b === 'compute').map(([id]) => id)).toEqual(['gemm-ideal']);
	});

	it('steps map to presets', () => {
		expect(LAST_STEP).toBe(PRESETS.length);
		expect(presetAt(0)).toBeNull();
		expect(presetAt(1)?.id).toBe('vadd');
		expect(presetAt(LAST_STEP)?.id).toBe('gemm-ideal');
	});
});

suite('status text', () => {
	it('describes every step', () => {
		for (let s = 0; s <= LAST_STEP; s++) expect(describe(cfg(), s).length).toBeGreaterThan(40);
		expect(describe(cfg(), 0)).toContain('memory-bound');
		expect(describe(cfg(), 5)).toContain('Zero FLOPs');
		expect(describe(cfg(), LAST_STEP)).toContain('compute-bound');
		expect(describe(cfg({ gpu: 't4', roof: 'int8' }), 0)).toContain('ops/byte');
	});
});
