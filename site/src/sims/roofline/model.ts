import { gpuById, type GpuSpec } from '@sim-core/gpu-specs';

/**
 * Roofline model (Williams, Waterman and Patterson, 2009): a kernel that does F
 * operations and moves B bytes to and from DRAM has arithmetic intensity
 * AI = F / B, and can run no faster than min(peak, AI x bandwidth). Peaks and
 * bandwidth come from gpu-specs.ts. Everything here is a model estimate.
 */
export const ROOF_IDS = ['fp64', 'fp32', 'tf32', 'fp16', 'fp8', 'int8'] as const;
export type RoofId = (typeof ROOF_IDS)[number];

export const ROOF_NAMES: Record<RoofId, string> = {
	fp64: 'FP64',
	fp32: 'FP32',
	tf32: 'TF32 tensor',
	fp16: 'FP16/BF16 tensor',
	fp8: 'FP8 tensor',
	int8: 'INT8 tensor',
};

/** Peak rate in TFLOPS (TOPS for INT8), or null where the GPU has no such mode. */
export function peakTflops(g: GpuSpec, r: RoofId): number | null {
	switch (r) {
		case 'fp64':
			return g.peakFp64Tflops;
		case 'fp32':
			return g.peakFp32Tflops;
		case 'tf32':
			return g.peakTf32TensorTflops;
		case 'fp16':
			return g.peakFp16TensorTflops;
		case 'fp8':
			return g.peakFp8TensorTflops;
		case 'int8':
			return g.peakInt8TensorTops;
	}
}

export const roofsOf = (g: GpuSpec) => ROOF_IDS.filter((r) => peakTflops(g, r) !== null);

/** DRAM bandwidth in TB/s, so that AI x bandwidth is in TFLOPS. */
export const bwTbPerS = (g: GpuSpec) => g.memoryGbPerS / 1000;

/** Arithmetic intensity at which the bandwidth roof meets the compute roof, FLOP/byte. */
export const ridge = (g: GpuSpec, r: RoofId) => (peakTflops(g, r) ?? 0) / bwTbPerS(g);

/** Attainable TFLOPS at a given intensity. */
export const attainable = (g: GpuSpec, r: RoofId, ai: number) => Math.min(peakTflops(g, r) ?? 0, ai * bwTbPerS(g));

/** Slider ranges, as powers of two. */
export const LOG2_FLOPS_MIN = 20;
export const LOG2_FLOPS_MAX = 48;
export const LOG2_BYTES_MIN = 20;
export const LOG2_BYTES_MAX = 38;
export const LOG2_STEP = 0.5;

/** Plot range: intensity 1/16 to 4096 FLOP/byte, 0.01 to 4000 TFLOPS. */
export const AI_MIN = 2 ** -4;
export const AI_MAX = 2 ** 12;
export const TF_MIN = 0.01;
export const TF_MAX = 4000;

export interface Config {
	gpu: GpuSpec['id'];
	roof: RoofId;
	log2Flops: number;
	log2Bytes: number;
}

export interface Point {
	ai: number;
	tflops: number;
	bound: 'memory' | 'compute';
	/** Attainable over the selected peak. */
	fractionOfPeak: number;
	/** Model time in microseconds: the slower of compute at peak and traffic at full bandwidth. */
	timeUs: number;
}

export function evaluate(g: GpuSpec, r: RoofId, flops: number, bytes: number): Point {
	const ai = flops / bytes;
	const peak = peakTflops(g, r) ?? 0;
	const tflops = attainable(g, r, ai);
	// 1 TFLOPS is 10^6 FLOPs per microsecond; 1 TB/s is 10^6 bytes per microsecond.
	const timeUs = Math.max(flops / (peak * 1e6), bytes / (bwTbPerS(g) * 1e6));
	return { ai, tflops, bound: ai < ridge(g, r) ? 'memory' : 'compute', fractionOfPeak: tflops / peak, timeUs };
}

export const kernelPoint = (c: Config) => evaluate(gpuById(c.gpu), c.roof, 2 ** c.log2Flops, 2 ** c.log2Bytes);

/** Roof to use on a GPU: the requested one if the GPU has it, else FP32. */
export const roofFor = (g: GpuSpec, r: RoofId): RoofId => (peakTflops(g, r) === null ? 'fp32' : r);

/**
 * Kernels from earlier chapters, FP32 data, counted per output element (per
 * input element for reduction and transpose) with every byte going to DRAM once.
 * GEMM is C = A x B with N x N matrices.
 */
export interface Preset {
	id: string;
	name: string;
	chapter: string;
	flops: number;
	bytes: number;
	how: string;
}

export const GEMM_N = 4096;
export const GEMM_TILE = 32;

export const PRESETS: Preset[] = [
	{ id: 'vadd', name: 'Vector add', chapter: '3', flops: 1, bytes: 12, how: 'c[i] = a[i] + b[i]: 1 add, 2 loads and 1 store of 4 bytes' },
	{ id: 'unfused', name: 'relu(a*x+y), three kernels', chapter: '12', flops: 3, bytes: 28, how: '3 operations, 7 arrays of 4 bytes through DRAM' },
	{ id: 'fused', name: 'relu(a*x+y), fused', chapter: '12', flops: 3, bytes: 12, how: '3 operations, x and y read, out written' },
	{ id: 'reduce', name: 'Sum reduction', chapter: '13', flops: 1, bytes: 4, how: 'one add per 4-byte element read' },
	{ id: 'transpose', name: 'Transpose', chapter: '11', flops: 0, bytes: 8, how: 'no arithmetic, 4 bytes read and 4 written per element' },
	{ id: 'gemm-naive', name: 'GEMM, no reuse', chapter: '20', flops: 2, bytes: 8, how: 'each multiply-add loads one A and one B value from DRAM' },
	{
		id: 'gemm-tiled',
		name: `GEMM, ${GEMM_TILE}x${GEMM_TILE} tiles`,
		chapter: '7, 20',
		flops: 2 * GEMM_TILE ** 3,
		bytes: 2 * GEMM_TILE ** 2 * 4,
		how: `each ${GEMM_TILE}x${GEMM_TILE} tile of A and of B is loaded once and used ${GEMM_TILE} times`,
	},
	{
		id: 'gemm-ideal',
		name: `GEMM, N = ${GEMM_N}, each matrix moved once`,
		chapter: '20',
		flops: 2 * GEMM_N ** 3,
		bytes: 3 * GEMM_N ** 2 * 4,
		how: '2N^3 FLOPs over reading A and B and writing C once',
	},
];

export const presetAi = (p: Preset) => p.flops / p.bytes;

/** Step 0 is the reader's kernel; step i highlights PRESETS[i - 1]. */
export const LAST_STEP = PRESETS.length;
export const presetAt = (step: number): Preset | null => PRESETS[step - 1] ?? null;

export function fmtNum(x: number): string {
	if (x === 0) return '0';
	if (x < 0.001) return x.toExponential(2);
	if (x >= 1000) return Math.round(x).toLocaleString('en-US');
	if (x >= 10) return String(+x.toFixed(1));
	return String(+x.toPrecision(3));
}

export function fmtPow2(log2: number, unit: string): string {
	const v = 2 ** log2;
	const units = ['', 'K', 'M', 'G', 'T', 'P'];
	let i = 0;
	let s = v;
	while (s >= 1000 && i < units.length - 1) {
		s /= 1000;
		i++;
	}
	return `${+s.toPrecision(3)} ${units[i]}${unit}`;
}

export function fmtUs(us: number): string {
	if (us >= 1e6) return `${+(us / 1e6).toPrecision(3)} s`;
	if (us >= 1000) return `${+(us / 1000).toPrecision(3)} ms`;
	if (us >= 1) return `${+us.toPrecision(3)} µs`;
	return `${+(us * 1000).toPrecision(3)} ns`;
}

const opUnit = (r: RoofId) => (r === 'int8' ? 'TOPS' : 'TFLOPS');
const aiUnit = (r: RoofId) => (r === 'int8' ? 'ops/byte' : 'FLOP/byte');

function verdict(g: GpuSpec, r: RoofId, p: Point): string {
	const peak = peakTflops(g, r)!;
	const rd = ridge(g, r);
	if (p.bound === 'memory') {
		return `Left of the ridge (${fmtNum(rd)} ${aiUnit(r)}): memory-bound. At most ${fmtNum(p.tflops)} ${opUnit(r)}, ${fmtNum(100 * p.fractionOfPeak)}% of the ${fmtNum(peak)} ${opUnit(r)} ${ROOF_NAMES[r]} peak; only fewer bytes per operation moves it up.`;
	}
	return `Right of the ridge (${fmtNum(rd)} ${aiUnit(r)}): compute-bound. Capped at the ${fmtNum(peak)} ${opUnit(r)} ${ROOF_NAMES[r]} peak; moving fewer bytes no longer helps.`;
}

export function describe(c: Config, step: number): string {
	const g = gpuById(c.gpu);
	const r = c.roof;
	const preset = presetAt(step);
	if (!preset) {
		const p = kernelPoint(c);
		return (
			`Your kernel on the ${g.name}: ${fmtPow2(c.log2Flops, 'FLOP')} over ${fmtPow2(c.log2Bytes, 'B')} is ${fmtNum(p.ai)} ${aiUnit(r)}. ` +
			`${verdict(g, r, p)} Model estimate ${fmtUs(p.timeUs)}.`
		);
	}
	const head = `Marker ${step}, ${preset.name} (chapter ${preset.chapter}): ${preset.how}.`;
	if (preset.flops === 0) {
		return `${head} Zero FLOPs, so it has no place on a FLOP roofline; judge it by GB/s against the ${g.memoryGbPerS.toLocaleString('en-US')} GB/s bandwidth roof.`;
	}
	const p = evaluate(g, r, preset.flops, preset.bytes);
	return `${head} ${fmtNum(p.ai)} ${aiUnit(r)}. ${verdict(g, r, p)}`;
}
