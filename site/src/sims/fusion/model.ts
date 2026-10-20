import { gpuById, type GpuSpec } from '@sim-core/gpu-specs';

/**
 * out = relu(a * x + y) on n float32 elements, computed two ways:
 * three elementwise kernels with two intermediate arrays in DRAM, or one fused
 * kernel that keeps the intermediates in registers.
 *
 * Every array element is read or written exactly once per kernel, so a kernel's
 * DRAM traffic is (arrays read + arrays written) x n x 4 bytes. Time is a model
 * estimate: that traffic at the preset's peak memory bandwidth (gpu-specs.ts),
 * plus a fixed cost per launch that the reader sets. NVIDIA's CUDA Graphs post
 * (Alan Gray, 2019, https://developer.nvidia.com/blog/cuda-graphs/) measured, on a
 * Tesla V100, a 2.9 us kernel taking 3.8 us per kernel when launched back to back,
 * so about 0.9 us of overhead each; the default below is rounded from that.
 */
export const FLOAT_BYTES = 4;
export const DEFAULT_LOG2N = 24;
export const LOG2N_MIN = 10;
export const LOG2N_MAX = 28;
export const DEFAULT_LAUNCH_US = 1;
export const LAUNCH_US_MAX = 10;

export const ARRAYS = ['x', 'y', 't', 'u', 'out'] as const;
export type ArrayId = (typeof ARRAYS)[number];
export const INTERMEDIATES: ArrayId[] = ['t', 'u'];

export interface Kernel {
	name: string;
	math: string;
	reads: ArrayId[];
	writes: ArrayId[];
}

export const PIPELINE_IDS = ['unfused', 'fused'] as const;
export type PipelineId = (typeof PIPELINE_IDS)[number];
export const PIPELINE_NAMES: Record<PipelineId, string> = { unfused: 'Three kernels', fused: 'One fused kernel' };

export const PIPELINES: Record<PipelineId, Kernel[]> = {
	unfused: [
		{ name: 'scale', math: 't = a * x', reads: ['x'], writes: ['t'] },
		{ name: 'add', math: 'u = t + y', reads: ['t', 'y'], writes: ['u'] },
		{ name: 'relu', math: 'out = max(u, 0)', reads: ['u'], writes: ['out'] },
	],
	fused: [{ name: 'fused', math: 'out = max(a * x + y, 0)', reads: ['x', 'y'], writes: ['out'] }],
};

/** Arrays each pipeline allocates in DRAM. */
export const allocated = (p: PipelineId): ArrayId[] => (p === 'fused' ? ['x', 'y', 'out'] : [...ARRAYS]);

export interface Config {
	gpu: GpuSpec['id'];
	log2n: number;
	/** Assumed fixed cost of one kernel launch, microseconds. */
	launchUs: number;
}

/** Steps: 0 start, 1..3 the unfused kernels in order (the fused one runs at step 1), 4 totals. */
export const LAST_STEP = 4;

/** Index of the kernel a pipeline runs at a step, or null. */
export function kernelAt(p: PipelineId, step: number): number | null {
	const i = step - 1;
	return i >= 0 && i < PIPELINES[p].length ? i : null;
}

export interface Totals {
	launches: number;
	readBytes: number;
	writeBytes: number;
	bytes: number;
	launchUs: number;
	memoryUs: number;
	timeUs: number;
}

export const arrayBytes = (n: number) => n * FLOAT_BYTES;

/** Traffic and modeled time of the kernels a pipeline has run by the end of `step`. */
export function totalsAt(c: Config, p: PipelineId, step: number): Totals {
	const g = gpuById(c.gpu);
	const n = 2 ** c.log2n;
	const done = PIPELINES[p].slice(0, Math.max(0, Math.min(step, LAST_STEP - 1)));
	const readBytes = done.reduce((s, k) => s + k.reads.length * arrayBytes(n), 0);
	const writeBytes = done.reduce((s, k) => s + k.writes.length * arrayBytes(n), 0);
	const bytes = readBytes + writeBytes;
	const launchUs = done.length * c.launchUs;
	// GB/s is 10^9 bytes per second, so bytes / (GB/s x 10^3) is microseconds.
	const memoryUs = bytes / (g.memoryGbPerS * 1e3);
	return { launches: done.length, readBytes, writeBytes, bytes, launchUs, memoryUs, timeUs: launchUs + memoryUs };
}

export const final = (c: Config, p: PipelineId) => totalsAt(c, p, LAST_STEP);

/** Modeled time of the three kernels over the fused one. */
export const speedup = (c: Config) => final(c, 'unfused').timeUs / final(c, 'fused').timeUs;

export function fmtBytes(b: number) {
	const units = ['bytes', 'KiB', 'MiB', 'GiB'];
	let i = 0;
	while (b >= 1024 && i < units.length - 1) {
		b /= 1024;
		i++;
	}
	return `${+b.toFixed(2)} ${units[i]}`;
}

export function fmtUs(us: number) {
	if (us === 0) return '0 µs';
	if (us >= 1000) return `${+(us / 1000).toPrecision(3)} ms`;
	if (us >= 1) return `${+us.toPrecision(3)} µs`;
	return `${+(us * 1000).toPrecision(3)} ns`;
}

export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

function describeKernel(c: Config, p: PipelineId, step: number) {
	const i = kernelAt(p, step);
	const n = 2 ** c.log2n;
	if (i === null) return `${PIPELINE_NAMES[p]}: done.`;
	const k = PIPELINES[p][i];
	const bytes = (k.reads.length + k.writes.length) * arrayBytes(n);
	const extra = p === 'fused' ? ' a * x + y stays in a register; nothing in between goes to DRAM.' : '';
	return `${PIPELINE_NAMES[p]}: launch ${k.name}, ${k.math}. Reads ${k.reads.join(' and ')}, writes ${k.writes.join(', ')}: ${fmtBytes(bytes)} of DRAM traffic.${extra}`;
}

export function describe(c: Config, step: number): string {
	const g = gpuById(c.gpu);
	const n = 2 ** c.log2n;
	if (step === 0) {
		return `out = relu(a * x + y) on ${fmt(n)} floats (${fmtBytes(arrayBytes(n))} per array). Guess how many bytes each version moves to and from DRAM, then step.`;
	}
	if (step < LAST_STEP) return PIPELINE_IDS.map((p) => describeKernel(c, p, step)).join(' ');
	const u = final(c, 'unfused');
	const f = final(c, 'fused');
	return (
		`Three kernels: ${fmtBytes(u.bytes)} moved, ${u.launches} launches, model estimate ${fmtUs(u.timeUs)}. ` +
		`One fused kernel: ${fmtBytes(f.bytes)} moved, 1 launch, model estimate ${fmtUs(f.timeUs)}. ` +
		`On the ${g.name} at ${fmt(g.memoryGbPerS)} GB/s with ${fmtUs(c.launchUs)} per launch, fusion is ${speedup(c).toFixed(2)}x faster in this model.`
	);
}

/** Source shown in the sim's code panel. */
export const CODE: Record<PipelineId, string> = {
	unfused: `__global__ void scale(const float *x, float *t, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) t[i] = a * x[i];
}
__global__ void add(const float *t, const float *y, float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = t[i] + y[i];
}
__global__ void relu(const float *u, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(u[i], 0.0f);
}

scale<<<blocks, threads>>>(x, t, a, n);
add<<<blocks, threads>>>(t, y, u, n);
relu<<<blocks, threads>>>(u, out, n);`,
	fused: `__global__ void fused(const float *x, const float *y, float *out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a * x[i] + y[i];
        out[i] = fmaxf(v, 0.0f);
    }
}

fused<<<blocks, threads>>>(x, y, out, a, n);`,
};

/** Highlighted lines (1-based) per step. */
export function linesAt(p: PipelineId, step: number): number[] {
	const i = kernelAt(p, step);
	if (i === null) return [];
	if (p === 'fused') return [4, 5, 9];
	return [[3, 14], [7, 15], [11, 16]][i];
}
