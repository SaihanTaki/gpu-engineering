import type { GpuSpec } from '@sim-core/gpu-specs';

/**
 * c = a + b on the GPU, step by step as the CUDA host code does it.
 * Times are lower bounds from peak bandwidth: PCIe copies at the per-direction
 * peak, the kernel at peak memory bandwidth. Fixed costs per call are left out.
 */
export const SIZES = [2 ** 10, 2 ** 16, 2 ** 20, 2 ** 24, 2 ** 28, 2 ** 30];
export const DEFAULT_N = 2 ** 24;
export const THREADS_PER_BLOCK = 256;
export const FLOAT_BYTES = 4;

export const CODE = `__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

size_t bytes = n * sizeof(float);  // h_a, h_b, h_c: host arrays
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, bytes);
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
int threads = ${THREADS_PER_BLOCK}, blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);`;

export type StepId = 'host' | 'malloc' | 'copyA' | 'copyB' | 'launch' | 'read' | 'write' | 'copyC' | 'free';

const STEPS: { id: StepId; lines: number[] }[] = [
	{ id: 'host', lines: [6] },
	{ id: 'malloc', lines: [7, 8, 9, 10] },
	{ id: 'copyA', lines: [11] },
	{ id: 'copyB', lines: [12] },
	{ id: 'launch', lines: [13, 14] },
	{ id: 'read', lines: [2, 3] },
	{ id: 'write', lines: [3] },
	{ id: 'copyC', lines: [15] },
	{ id: 'free', lines: [16] },
];
export const lastStep = STEPS.length - 1;

export const arrayBytes = (n: number) => n * FLOAT_BYTES;
export const blocksFor = (n: number) => Math.ceil(n / THREADS_PER_BLOCK);

/** Seconds, from peak bandwidth in GB/s (10^9 bytes per second). */
export function times(g: GpuSpec, n: number) {
	const pcie = g.pcieGbPerS * 1e9;
	const copyIn = (2 * arrayBytes(n)) / pcie;
	const kernel = (3 * arrayBytes(n)) / (g.memoryGbPerS * 1e9);
	const copyOut = arrayBytes(n) / pcie;
	return { copyIn, kernel, copyOut, total: copyIn + kernel + copyOut };
}

/** 'none': not allocated. 'empty': allocated, no data yet. 'full': holds the array. */
export type BufState = 'none' | 'empty' | 'full';
export type BufId = 'hA' | 'hB' | 'hC' | 'dA' | 'dB' | 'dC';
export type SmState = 'idle' | 'assigned' | 'read' | 'write' | 'done';

export interface State {
	id: StepId;
	lines: number[];
	buffers: Record<BufId, BufState>;
	/** The PCIe copy in flight during this step, if any. */
	transfer: { from: BufId; to: BufId } | null;
	sm: SmState;
	/** Seconds of each phase finished by the end of this step. */
	done: { copyIn: number; kernel: number; copyOut: number };
}

const clampStep = (step: number) => Math.min(Math.max(0, Math.round(step)), lastStep);

export function stateAt(g: GpuSpec, n: number, step: number): State {
	const s = clampStep(step);
	const { id, lines } = STEPS[s];
	const t = times(g, n);
	const after = (k: StepId) => s >= STEPS.findIndex((x) => x.id === k);
	const dev = (filledAt: StepId): BufState => (s === lastStep ? 'none' : after(filledAt) ? 'full' : after('malloc') ? 'empty' : 'none');
	return {
		id,
		lines,
		buffers: {
			hA: 'full',
			hB: 'full',
			hC: after('copyC') ? 'full' : 'empty',
			dA: dev('copyA'),
			dB: dev('copyB'),
			dC: dev('write'),
		},
		transfer: id === 'copyA' ? { from: 'hA', to: 'dA' } : id === 'copyB' ? { from: 'hB', to: 'dB' } : id === 'copyC' ? { from: 'dC', to: 'hC' } : null,
		sm: id === 'launch' ? 'assigned' : id === 'read' ? 'read' : id === 'write' ? 'write' : after('write') ? 'done' : 'idle',
		done: {
			copyIn: after('copyB') ? t.copyIn : after('copyA') ? t.copyIn / 2 : 0,
			// Reads are 2 of the kernel's 3 arrays of traffic.
			kernel: after('write') ? t.kernel : after('read') ? (t.kernel * 2) / 3 : 0,
			copyOut: after('copyC') ? t.copyOut : 0,
		},
	};
}

export const fmt = (n: number) => n.toLocaleString('en-US');

/** Sizes in powers of 1024; every size here is a power of two times 1, 2 or 3. */
export function fmtBytes(b: number) {
	const units = ['bytes', 'KiB', 'MiB', 'GiB'];
	let i = 0;
	while (b >= 1024 && i < units.length - 1) {
		b /= 1024;
		i++;
	}
	return `${+b.toFixed(2)} ${units[i]}`;
}

export function fmtTime(s: number) {
	if (s === 0) return '0';
	const [scale, unit] = s >= 1 ? [1, 's'] : s >= 1e-3 ? [1e3, 'ms'] : s >= 1e-6 ? [1e6, 'µs'] : [1e9, 'ns'];
	return `${(s * scale).toPrecision(3)} ${unit}`;
}

export const pct = (part: number, whole: number) => `${((100 * part) / whole).toFixed(1)}%`;

export function describe(g: GpuSpec, n: number, step: number): string {
	const st = stateAt(g, n, step);
	const t = times(g, n);
	const size = fmtBytes(arrayBytes(n));
	const link = `PCIe Gen${g.pcieGen} x16 (${g.pcieGbPerS} GB/s each way, peak)`;
	const blocks = blocksFor(n);
	switch (st.id) {
		case 'host':
			return `h_a and h_b each hold ${fmt(n)} floats (${size}) in CPU RAM; h_c is allocated but empty. The kernel reads its inputs from GPU memory, so they must be copied there first. Predict: will the copies or the kernel take longer? Then press Next.`;
		case 'malloc':
			return `cudaMalloc reserves three buffers of ${size} in the GPU's ${g.memory}: d_a, d_b and d_c. They hold no data yet. Allocation time is left out of the time bar.`;
		case 'copyA':
			return `cudaMemcpy copies h_a to d_a over ${link}: ${size} takes at least ${fmtTime(t.copyIn / 2)}.`;
		case 'copyB':
			return `cudaMemcpy copies h_b to d_b: another ${fmtTime(t.copyIn / 2)}. Copying both inputs in takes at least ${fmtTime(t.copyIn)}.`;
		case 'launch': {
			const spread =
				blocks < g.sms
					? `Only ${fmt(blocks)} of the ${g.sms} SMs get a block; the rest sit idle.`
					: `The ${g.sms} SMs take new blocks as earlier ones finish, about ${fmt(Math.round(blocks / g.sms))} per SM in total.`;
			return `vector_add<<<${fmt(blocks)}, ${THREADS_PER_BLOCK}>>> launches ${fmt(blocks)} block${blocks === 1 ? '' : 's'} of ${THREADS_PER_BLOCK} threads, one thread per element. ${spread} The launch returns to the CPU at once; the GPU runs the kernel on its own.`;
		}
		case 'read':
			return `Each thread i reads a[i] and b[i] from ${g.memory}: ${fmtBytes(2 * arrayBytes(n))} in all. At ${fmt(g.memoryGbPerS)} GB/s that takes at least ${fmtTime((t.kernel * 2) / 3)}.`;
		case 'write':
			return `Each thread adds its two values in registers and writes c[i] to d_c: ${size} more. The whole kernel moves ${fmtBytes(3 * arrayBytes(n))} and takes at least ${fmtTime(t.kernel)}.`;
		case 'copyC':
			return `cudaMemcpy waits for the kernel to finish, then copies d_c back to h_c over PCIe: ${size} takes at least ${fmtTime(t.copyOut)}. h_c now holds the result.`;
		case 'free': {
			const copies = t.copyIn + t.copyOut;
			return `cudaFree releases d_a, d_b and d_c. Total at least ${fmtTime(t.total)}: copies ${fmtTime(copies)} (${pct(copies, t.total)}), kernel ${fmtTime(t.kernel)} (${pct(t.kernel, t.total)}). The copies and the kernel both move ${fmtBytes(3 * arrayBytes(n))}, but PCIe is ${(g.memoryGbPerS / g.pcieGbPerS).toFixed(1)}x slower than ${g.memory}, so for vector add the copies cost more than the work.`;
		}
	}
}
