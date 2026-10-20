import type { GpuSpec } from '@sim-core/gpu-specs';

/**
 * x = 2 * x on the GPU, with the CPU writing x before the kernel and reading it after.
 * Four ways to get x across PCIe: managed memory moved by page faults, managed memory
 * moved by cudaMemPrefetchAsync, and explicit copies from pageable or pinned host memory.
 *
 * Transfer times are lower bounds from the per-direction PCIe peak in gpu-specs.ts.
 * Three things NVIDIA does not publish a number for are model assumptions:
 *   - one GPU fault migrates one whole 2 MiB page (the page size is NVIDIA's, the granularity is ours),
 *   - each fault adds a fixed handling cost, picked by the reader,
 *   - the staging copy for pageable memory runs at the PCIe rate and does not overlap the DMA.
 */
export const MIB = 1024 * 1024;
/** GPU-resident page size for cudaMallocManaged, from the programming guide's allocator table. */
export const PAGE_BYTES = 2 * MIB;
export const PAGE_COUNTS = [8, 16];
export const DEFAULT_PAGES = 8;
/** Assumed fault handling cost in microseconds; NVIDIA does not publish one. */
export const FAULT_US = [0, 10, 20, 50];
export const DEFAULT_FAULT_US = 20;

export type Mode = 'demand' | 'prefetch' | 'pageable' | 'pinned';
export const MODES: { id: Mode; label: string }[] = [
	{ id: 'demand', label: 'Managed, on-demand' },
	{ id: 'prefetch', label: 'Managed + prefetch' },
	{ id: 'pageable', label: 'cudaMemcpy, pageable' },
	{ id: 'pinned', label: 'cudaMemcpy, pinned' },
];

const KERNEL = `__global__ void scale(float *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.0f * x[i];
}
`;

export const CODE: Record<Mode, string> = {
	demand: `${KERNEL}
float *x;
cudaMallocManaged(&x, bytes);
for (size_t i = 0; i < n; i++) x[i] = i;
scale<<<blocks, 256>>>(x, n);
cudaDeviceSynchronize();
for (size_t i = 0; i < n; i++) sum += x[i];
cudaFree(x);`,
	prefetch: `${KERNEL}
float *x;
cudaMallocManaged(&x, bytes);
for (size_t i = 0; i < n; i++) x[i] = i;
cudaMemLocation gpu = {}, host = {};
gpu.type = cudaMemLocationTypeDevice; gpu.id = 0;
host.type = cudaMemLocationTypeHost;
cudaMemPrefetchAsync(x, bytes, gpu, 0, stream);
scale<<<blocks, 256, 0, stream>>>(x, n);
cudaMemPrefetchAsync(x, bytes, host, 0, stream);
cudaStreamSynchronize(stream);
for (size_t i = 0; i < n; i++) sum += x[i];
cudaFree(x);`,
	pageable: `${KERNEL}
float *h_x = (float *)malloc(bytes), *d_x;
cudaMalloc(&d_x, bytes);
for (size_t i = 0; i < n; i++) h_x[i] = i;
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
scale<<<blocks, 256>>>(d_x, n);
cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
for (size_t i = 0; i < n; i++) sum += h_x[i];
free(h_x); cudaFree(d_x);`,
	pinned: `${KERNEL}
float *h_x, *d_x;
cudaMallocHost(&h_x, bytes);
cudaMalloc(&d_x, bytes);
for (size_t i = 0; i < n; i++) h_x[i] = i;
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
scale<<<blocks, 256>>>(d_x, n);
cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
for (size_t i = 0; i < n; i++) sum += h_x[i];
cudaFreeHost(h_x); cudaFree(d_x);`,
};

/** 'none': no memory here. 'empty': allocated, no data. 'x': the input. '2x': the result. */
export type Cell = 'none' | 'empty' | 'x' | '2x';

export type StepId =
	| 'alloc'
	| 'init'
	| 'launch'
	| 'gpuFault'
	| 'sync'
	| 'cpuFault'
	| 'prefetchIn'
	| 'kernel'
	| 'prefetchOut'
	| 'read'
	| 'stageIn'
	| 'copyIn'
	| 'copyOut'
	| 'stageOut'
	| 'free';

export interface Cost {
	pcie: number;
	fault: number;
	staging: number;
	kernel: number;
}

export interface State {
	id: StepId;
	lines: number[];
	/** Page k of the array on each side. */
	cpu: Cell[];
	gpu: Cell[];
	/** The driver's pinned staging buffer, used only for pageable copies. */
	staging: Cell;
	/** Pages crossing PCIe in this step. */
	transfer: { dir: 'h2d' | 'd2h'; pages: number[] } | null;
	/** The page that faulted in this step, and on which side. */
	fault: { side: 'gpu' | 'cpu'; page: number } | null;
	gpuFaults: number;
	cpuFaults: number;
	/** Separate transfers started over PCIe so far. */
	transfers: number;
	bytes: number;
	/** Seconds spent so far, by kind. */
	done: Cost;
	/** Seconds this step added. */
	added: Cost;
}

export interface Params {
	gpu: GpuSpec;
	mode: Mode;
	pages: number;
	faultUs: number;
}

const line = (mode: Mode, text: string) => {
	const i = CODE[mode].split('\n').findIndex((l) => l.includes(text));
	if (i < 0) throw new Error(`no line with ${text}`);
	return i + 1;
};

const zero = (): Cost => ({ pcie: 0, fault: 0, staging: 0, kernel: 0 });
export const total = (c: Cost) => c.pcie + c.fault + c.staging + c.kernel;

export const pcieSeconds = (g: GpuSpec, bytes: number) => bytes / (g.pcieGbPerS * 1e9);
/** The kernel reads and writes every byte once. */
export const kernelSeconds = (g: GpuSpec, bytes: number) => (2 * bytes) / (g.memoryGbPerS * 1e9);

/** Every state of one run, step 0 first. */
export function run(p: Params): State[] {
	const { gpu: g, mode, pages, faultUs } = p;
	const bytes = pages * PAGE_BYTES;
	const all = [...Array(pages).keys()];
	const managed = mode === 'demand' || mode === 'prefetch';
	const states: State[] = [];
	let cur: State = {
		id: 'alloc',
		lines: [],
		cpu: all.map(() => (managed ? 'none' : 'empty')),
		gpu: all.map(() => (managed ? 'none' : 'empty')),
		staging: 'none',
		transfer: null,
		fault: null,
		gpuFaults: 0,
		cpuFaults: 0,
		transfers: 0,
		bytes: 0,
		done: zero(),
		added: zero(),
	};

	const push = (id: StepId, lines: string[], change: (s: State) => void, added: Partial<Cost> = {}) => {
		const s: State = {
			...cur,
			id,
			lines: lines.map((t) => line(mode, t)),
			cpu: [...cur.cpu],
			gpu: [...cur.gpu],
			transfer: null,
			fault: null,
			added: { ...zero(), ...added },
		};
		change(s);
		s.done = { pcie: cur.done.pcie, fault: cur.done.fault, staging: cur.done.staging, kernel: cur.done.kernel };
		for (const k of Object.keys(s.added) as (keyof Cost)[]) s.done[k] += s.added[k];
		states.push(s);
		cur = s;
	};
	const move = (s: State, dir: 'h2d' | 'd2h', pg: number[]) => {
		s.transfer = { dir, pages: pg };
		s.transfers += 1;
		s.bytes += pg.length * PAGE_BYTES;
	};
	const fill = (cells: Cell[], v: Cell) => cells.fill(v);

	const kernelT = kernelSeconds(g, bytes);
	const pageT = pcieSeconds(g, PAGE_BYTES);
	const bulkT = pcieSeconds(g, bytes);
	const faultT = faultUs * 1e-6;

	if (mode === 'demand') {
		push('alloc', ['cudaMallocManaged'], () => {});
		push('init', ['x[i] = i;'], (s) => fill(s.cpu, 'x'));
		push('launch', ['scale<<<'], () => {});
		for (const k of all)
			push(
				'gpuFault',
				['if (i < n)'],
				(s) => {
					s.fault = { side: 'gpu', page: k };
					s.gpuFaults += 1;
					move(s, 'h2d', [k]);
					s.cpu[k] = 'none';
					s.gpu[k] = '2x';
				},
				{ pcie: pageT, fault: faultT, kernel: kernelT / pages },
			);
		push('sync', ['cudaDeviceSynchronize'], () => {});
		for (const k of all)
			push(
				'cpuFault',
				['sum += x[i]'],
				(s) => {
					s.fault = { side: 'cpu', page: k };
					s.cpuFaults += 1;
					move(s, 'd2h', [k]);
					s.gpu[k] = 'none';
					s.cpu[k] = '2x';
				},
				{ pcie: pageT, fault: faultT },
			);
		push('free', ['cudaFree(x)'], (s) => fill(s.cpu, 'none'));
	} else if (mode === 'prefetch') {
		push('alloc', ['cudaMallocManaged'], () => {});
		push('init', ['x[i] = i;'], (s) => fill(s.cpu, 'x'));
		push(
			'prefetchIn',
			['cudaMemLocation gpu', 'gpu.type', 'x, bytes, gpu'],
			(s) => {
				move(s, 'h2d', all);
				fill(s.cpu, 'none');
				fill(s.gpu, 'x');
			},
			{ pcie: bulkT },
		);
		push('kernel', ['scale<<<', 'if (i < n)'], (s) => fill(s.gpu, '2x'), { kernel: kernelT });
		push(
			'prefetchOut',
			['host.type', 'x, bytes, host'],
			(s) => {
				move(s, 'd2h', all);
				fill(s.gpu, 'none');
				fill(s.cpu, '2x');
			},
			{ pcie: bulkT },
		);
		push('read', ['cudaStreamSynchronize', 'sum += x[i]'], () => {});
		push('free', ['cudaFree(x)'], (s) => fill(s.cpu, 'none'));
	} else {
		const pageable = mode === 'pageable';
		push('alloc', pageable ? ['malloc(bytes)', 'cudaMalloc('] : ['cudaMallocHost', 'cudaMalloc('], () => {});
		push('init', ['h_x[i] = i;'], (s) => fill(s.cpu, 'x'));
		if (pageable) push('stageIn', ['cudaMemcpyHostToDevice'], (s) => (s.staging = 'x'), { staging: bulkT });
		push(
			'copyIn',
			['cudaMemcpyHostToDevice'],
			(s) => {
				move(s, 'h2d', all);
				fill(s.gpu, 'x');
				if (pageable) s.staging = 'empty';
			},
			{ pcie: bulkT },
		);
		push('kernel', ['scale<<<', 'if (i < n)'], (s) => fill(s.gpu, '2x'), { kernel: kernelT });
		push(
			'copyOut',
			['cudaMemcpyDeviceToHost'],
			(s) => {
				move(s, 'd2h', all);
				if (pageable) s.staging = '2x';
				else fill(s.cpu, '2x');
			},
			{ pcie: bulkT },
		);
		if (pageable)
			push(
				'stageOut',
				['cudaMemcpyDeviceToHost'],
				(s) => {
					s.staging = 'empty';
					fill(s.cpu, '2x');
				},
				{ staging: bulkT },
			);
		push('read', ['sum += h_x[i]'], () => {});
		push('free', [pageable ? 'free(h_x)' : 'cudaFreeHost'], (s) => {
			fill(s.cpu, 'none');
			fill(s.gpu, 'none');
			s.staging = 'none';
		});
	}
	return states;
}

export const lastStep = (p: Params) => run(p).length - 1;

export function stateAt(p: Params, step: number): State {
	const states = run(p);
	return states[Math.min(Math.max(0, Math.round(step)), states.length - 1)];
}

/** Totals for a whole run, for the comparison table and the exercises. */
export function summary(p: Params) {
	const end = run(p).at(-1)!;
	return { ...end.done, total: total(end.done), gpuFaults: end.gpuFaults, cpuFaults: end.cpuFaults, transfers: end.transfers, bytes: end.bytes };
}

export const fmt = (n: number) => n.toLocaleString('en-US');
export const fmtMiB = (b: number) => `${+(b / MIB).toFixed(2)} MiB`;

export function fmtTime(s: number) {
	if (s === 0) return '0';
	const [scale, unit] = s >= 1 ? [1, 's'] : s >= 1e-3 ? [1e3, 'ms'] : s >= 1e-6 ? [1e6, 'µs'] : [1e9, 'ns'];
	return `${(s * scale).toPrecision(3)} ${unit}`;
}

const plural = (n: number, w: string) => `${fmt(n)} ${w}${n === 1 ? '' : 's'}`;

export function describe(p: Params, step: number): string {
	const s = stateAt(p, step);
	const { gpu: g, pages, faultUs } = p;
	const bytes = pages * PAGE_BYTES;
	const size = fmtMiB(bytes);
	const page = s.fault ? s.fault.page : 0;
	const pageT = fmtTime(pcieSeconds(g, PAGE_BYTES));
	const bulkT = fmtTime(pcieSeconds(g, bytes));
	const link = `PCIe Gen${g.pcieGen} (${g.pcieGbPerS} GB/s each way, peak)`;
	switch (s.id) {
		case 'alloc':
			if (p.mode === 'demand' || p.mode === 'prefetch')
				return `cudaMallocManaged reserves ${size} of addresses, ${plural(pages, 'page')} of 2 MiB, valid on both CPU and GPU. No page has physical memory yet: a page is placed where it is first touched. Predict: how many page faults will this run take?`;
			return p.mode === 'pageable'
				? `malloc returns ${size} of ordinary pageable memory, which the OS may swap out or move. cudaMalloc reserves ${size} in the GPU's ${g.memory}. Two buffers, two pointers.`
				: `cudaMallocHost returns ${size} of pinned (page-locked) host memory: the OS will not swap or move it, so the GPU's copy engine can read it directly. cudaMalloc reserves ${size} on the GPU.`;
		case 'init':
			return p.mode === 'demand' || p.mode === 'prefetch'
				? `The CPU writes x[i] = i. Each first write populates a page in CPU RAM, so all ${plural(pages, 'page')} now live on the CPU.`
				: `The CPU fills h_x. The GPU buffer is still empty.`;
		case 'launch':
			return `The kernel launches with no data on the GPU. Its first access to each page will fault.`;
		case 'gpuFault':
			return `GPU page fault on page ${page}: a warp touches an address whose page is in CPU RAM. The driver handles the fault (assumed ${faultUs} µs), moves the 2 MiB page over ${link} in at least ${pageT}, and the threads resume and write 2x. GPU faults so far: ${s.gpuFaults}.`;
		case 'sync':
			return `The kernel has finished after ${plural(s.gpuFaults, 'fault')} and ${plural(s.transfers, 'separate transfer')}. Every page now lives on the GPU.`;
		case 'cpuFault':
			return `CPU page fault on page ${page}: the host loop reads an address whose page is on the GPU, so the page moves back over PCIe (at least ${pageT}, plus the assumed ${faultUs} µs). CPU faults so far: ${s.cpuFaults}.`;
		case 'prefetchIn':
			return `cudaMemPrefetchAsync moves all ${size} to the GPU in one bulk transfer, at least ${bulkT}, before the kernel starts. It is queued in the stream, so the kernel waits for it.`;
		case 'kernel':
			return `The kernel reads and writes x in the GPU's ${g.memory}: ${fmtMiB(2 * bytes)} of traffic at ${fmt(g.memoryGbPerS)} GB/s, at least ${fmtTime(kernelSeconds(g, bytes))}. No faults: every page is already there.`;
		case 'prefetchOut':
			return `A second cudaMemPrefetchAsync, to cudaMemLocationTypeHost, moves the result back in one transfer, at least ${bulkT}.`;
		case 'read':
			return p.mode === 'prefetch'
				? 'cudaStreamSynchronize waits for the kernel and the prefetch back. The CPU then reads the result with no faults: every page is already in CPU RAM.'
				: 'cudaMemcpy has returned, so h_x holds the result. The CPU reads it from its own memory.';
		case 'stageIn':
			return `h_x is pageable, so the copy engine cannot read it directly. The driver first copies it into a pinned staging buffer with the CPU. The model charges this at the PCIe rate with no overlap (an assumption): ${bulkT}.`;
		case 'copyIn':
			return p.mode === 'pageable'
				? `The copy engine sends the staging buffer to d_x over ${link}: at least ${bulkT}.`
				: `h_x is pinned, so the copy engine reads it directly and sends it to d_x over ${link} in one DMA: at least ${bulkT}.`;
		case 'copyOut':
			return p.mode === 'pageable'
				? `cudaMemcpy back: the DMA writes the result into the staging buffer, at least ${bulkT}.`
				: `cudaMemcpy back: one DMA straight into h_x, at least ${bulkT}.`;
		case 'stageOut':
			return `The CPU copies the staging buffer into h_x, another ${bulkT} in the model. Only now does cudaMemcpy return.`;
		case 'free': {
			const sm = summary(p);
			return `Freed. Total ${fmtTime(sm.total)}: PCIe ${fmtTime(sm.pcie)}, faults ${fmtTime(sm.fault)}, staging ${fmtTime(sm.staging)}, kernel ${fmtTime(sm.kernel)}. ${plural(sm.gpuFaults, 'GPU fault')}, ${plural(sm.cpuFaults, 'CPU fault')}, ${plural(sm.transfers, 'transfer')}.`;
		}
	}
}
