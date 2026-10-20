/**
 * One small sparse matrix, its COO, CSR and ELL arrays, and how three SpMV
 * kernels from Bell and Garland, "Efficient Sparse Matrix-Vector Multiplication
 * on CUDA" (NVIDIA Technical Report NVR-2008-004, 2008), hand its rows to threads:
 * CSR scalar (thread per row), CSR vector (warp per row) and ELL (thread per row,
 * padded, column-major).
 *
 * Constants: a warp is 32 threads (CUDA C++ Programming Guide, "SIMT Architecture").
 * Global memory requests are served in 32-byte sectors (CUDA C++ Best Practices Guide,
 * "Coalesced Access to Global Memory"); col and val entries are 4 bytes, so a sector
 * holds 8 of them. Arrays start at sector boundaries.
 */
export const N = 12;
export const WARP = 32;
export const ENTRY_BYTES = 4;
export const SECTOR_BYTES = 32;
export const PER_SECTOR = SECTOR_BYTES / ENTRY_BYTES;
/** __shfl_down_sync offsets 16, 8, 4, 2, 1. */
export const SHUFFLES = Math.log2(WARP);

export const PRESET_IDS = ['banded', 'random', 'power-law'] as const;
export type PresetId = (typeof PRESET_IDS)[number];
export const PRESET_NAMES: Record<PresetId, string> = {
	banded: 'Banded (tridiagonal)',
	random: 'Random rows',
	'power-law': 'Power-law rows',
};

export const KERNEL_IDS = ['scalar', 'vector', 'ell'] as const;
export type KernelId = (typeof KERNEL_IDS)[number];
export const KERNEL_NAMES: Record<KernelId, string> = {
	scalar: 'CSR scalar',
	vector: 'CSR vector',
	ell: 'ELL',
};

export const FORMAT_IDS = ['coo', 'csr', 'ell'] as const;
export type FormatId = (typeof FORMAT_IDS)[number];
export const FORMAT_NAMES: Record<FormatId, string> = { coo: 'COO', csr: 'CSR', ell: 'ELL' };
export const kernelFormat = (k: KernelId): FormatId => (k === 'ell' ? 'ell' : 'csr');

/** Random rows: lengths 1 to 5, columns picked by hand. */
const RANDOM: number[][] = [
	[1, 7],
	[0, 4, 9],
	[3],
	[2, 5, 8, 11],
	[6, 10],
	[0, 3, 7],
	[1, 5, 9, 10, 11],
	[4, 8],
	[2, 6, 11],
	[9],
	[0, 3, 5, 10],
	[7, 8],
];

/** Sorted column indices of the nonzeros in each row. */
export function pattern(p: PresetId): number[][] {
	if (p === 'random') return RANDOM.map((r) => [...r]);
	return Array.from({ length: N }, (_, r) => {
		if (p === 'banded') return [r - 1, r, r + 1].filter((c) => c >= 0 && c < N);
		// Row r has ceil(N / (r + 1)) nonzeros: one full row, a few medium ones, many short ones.
		const len = Math.ceil(N / (r + 1));
		const cols = new Set(Array.from({ length: len }, (_, j) => (r * 7 + j * 5) % N));
		return [...cols].sort((a, b) => a - b);
	});
}

/** Small integer values, so the arrays can print them. */
export const value = (r: number, c: number) => 1 + ((3 * r + 5 * c) % 9);

export interface Csr {
	rowPtr: number[];
	col: number[];
	val: number[];
	/** COO row array: the row of every entry. */
	row: number[];
}

export function csr(p: PresetId): Csr {
	const rows = pattern(p);
	const out: Csr = { rowPtr: [0], col: [], val: [], row: [] };
	rows.forEach((cols, r) => {
		for (const c of cols) {
			out.col.push(c);
			out.val.push(value(r, c));
			out.row.push(r);
		}
		out.rowPtr.push(out.col.length);
	});
	return out;
}

export interface Ell {
	width: number;
	/** Column-major: slot k of row r at k * N + r. Padding has col 0, val 0 and pad true. */
	col: number[];
	val: number[];
	pad: boolean[];
}

export function ell(p: PresetId): Ell {
	const rows = pattern(p);
	const width = Math.max(...rows.map((r) => r.length));
	const out: Ell = { width, col: [], val: [], pad: [] };
	for (let k = 0; k < width; k++) {
		for (let r = 0; r < N; r++) {
			const c = rows[r][k];
			out.col.push(c ?? 0);
			out.val.push(c === undefined ? 0 : value(r, c));
			out.pad.push(c === undefined);
		}
	}
	return out;
}

export const lengths = (p: PresetId) => pattern(p).map((r) => r.length);
export const nnz = (p: PresetId) => lengths(p).reduce((a, b) => a + b, 0);
export const maxLen = (p: PresetId) => Math.max(...lengths(p));
/** Loop passes of the vector kernel: the longest row in strips of 32. */
export const vectorPasses = (p: PresetId) => Math.max(...lengths(p).map((l) => Math.ceil(l / WARP)));

/** Loop iterations of the slowest thread (or warp), before y is written. */
export function loopSteps(k: KernelId, p: PresetId) {
	if (k === 'vector') return vectorPasses(p) + SHUFFLES;
	return maxLen(p);
}

/** Step 0 is the start; 1..loopSteps run the loop; the last step writes y and shows totals. */
export const lastStep = (k: KernelId, p: PresetId) => loopSteps(k, p) + 1;

export interface Read {
	row: number;
	/** Thread index within its warp: the row for scalar and ELL, the lane for vector. */
	lane: number;
	/** Warp that issues the read. */
	warp: number;
	/** Entry in the CSR (and COO) col and val arrays; null for ELL padding. */
	csrIndex: number | null;
	/** Slot k of the row, its column in the ELL picture. */
	slot: number;
	/** Address in the arrays this kernel reads, in entries. */
	address: number;
	pad: boolean;
}

export type Phase = { kind: 'start' } | { kind: 'loop'; iter: number } | { kind: 'shuffle'; offset: number } | { kind: 'write' };

export function phaseAt(k: KernelId, p: PresetId, step: number): Phase {
	if (step <= 0) return { kind: 'start' };
	if (step >= lastStep(k, p)) return { kind: 'write' };
	if (k === 'vector' && step > vectorPasses(p)) return { kind: 'shuffle', offset: WARP >> (step - vectorPasses(p)) };
	return { kind: 'loop', iter: step - 1 };
}

/** The col and val reads every thread makes at this step (the same addresses in both arrays). */
export function readsAt(k: KernelId, p: PresetId, step: number): Read[] {
	const ph = phaseAt(k, p, step);
	if (ph.kind !== 'loop') return [];
	const { rowPtr } = csr(p);
	const lens = lengths(p);
	const out: Read[] = [];
	for (let r = 0; r < N; r++) {
		if (k === 'vector') {
			for (let lane = 0; lane < WARP; lane++) {
				const slot = ph.iter * WARP + lane;
				if (slot >= lens[r]) break;
				const i = rowPtr[r] + slot;
				out.push({ row: r, lane, warp: r, csrIndex: i, slot, address: i, pad: false });
			}
		} else if (k === 'scalar') {
			if (ph.iter < lens[r]) {
				const i = rowPtr[r] + ph.iter;
				out.push({ row: r, lane: r % WARP, warp: Math.floor(r / WARP), csrIndex: i, slot: ph.iter, address: i, pad: false });
			}
		} else {
			const pad = ph.iter >= lens[r];
			const csrIndex = pad ? null : rowPtr[r] + ph.iter;
			out.push({ row: r, lane: r % WARP, warp: Math.floor(r / WARP), csrIndex, slot: ph.iter, address: ph.iter * N + r, pad });
		}
	}
	return out;
}

/** Sectors one array's reads touch at this step, counted per warp request and summed. */
export function sectorsAt(k: KernelId, p: PresetId, step: number) {
	const byWarp = new Map<number, Set<number>>();
	for (const r of readsAt(k, p, step)) {
		const s = byWarp.get(r.warp) ?? new Set<number>();
		s.add(Math.floor(r.address / PER_SECTOR));
		byWarp.set(r.warp, s);
	}
	let n = 0;
	for (const s of byWarp.values()) n += s.size;
	return n;
}

export interface Totals {
	/** Loop iterations of the slowest thread, shuffles included. */
	steps: number;
	/** Multiplies by a stored nonzero. */
	useful: number;
	/** Thread slots the loop occupies: threads x loop iterations. */
	slots: number;
	/** ELL padding slots: loaded and multiplied, but always 0. */
	padding: number;
	/** Slots where a thread has no entry to read and waits. */
	idle: number;
	/** Sectors read from col, and the same again from val. */
	sectors: number;
}

export function totals(k: KernelId, p: PresetId): Totals {
	const useful = nnz(p);
	let sectors = 0;
	for (let s = 1; s < lastStep(k, p); s++) sectors += sectorsAt(k, p, s);
	if (k === 'vector') {
		const slots = N * WARP * vectorPasses(p);
		return { steps: loopSteps(k, p), useful, slots, padding: 0, idle: slots - useful, sectors };
	}
	const slots = N * maxLen(p);
	const waste = slots - useful;
	return { steps: loopSteps(k, p), useful, slots, padding: k === 'ell' ? waste : 0, idle: k === 'ell' ? 0 : waste, sectors };
}

export const pct = (a: number, b: number) => Math.round((100 * a) / b);
const plural = (n: number, w: string, ws = `${w}s`) => `${n} ${n === 1 ? w : ws}`;

export function describe(k: KernelId, p: PresetId, step: number): string {
	const name = KERNEL_NAMES[k];
	const ph = phaseAt(k, p, step);
	const lens = lengths(p);
	if (ph.kind === 'start') {
		const who = k === 'vector' ? `${N} warps, one per row, ${N * WARP} lanes in all` : `${N} threads of one warp, one per row`;
		return `${name} on the ${PRESET_NAMES[p].toLowerCase()} matrix: ${nnz(p)} nonzeros in ${N} rows of ${Math.min(...lens)} to ${maxLen(p)}. ${who}. Guess how many steps the slowest thread needs, then step.`;
	}
	if (ph.kind === 'write') {
		const t = totals(k, p);
		const waste = k === 'ell' ? `${t.padding} padding slots multiply 0` : `${t.idle} slots idle`;
		return `${name} writes y. ${plural(t.steps, 'step')}; ${t.useful} of ${t.slots} thread slots did useful work (${pct(t.useful, t.slots)}%), ${waste}. Reads of col took ${plural(t.sectors, 'sector')}, and val the same.`;
	}
	if (ph.kind === 'shuffle') {
		return `${name}: shuffle with offset ${ph.offset}. In each warp, lane i adds the partial sum of lane i + ${ph.offset}${ph.offset === 1 ? ', so lane 0 now holds the row sum' : ''}. No memory is read.`;
	}
	const reads = readsAt(k, p, step);
	const sectors = plural(sectorsAt(k, p, step), 'sector');
	if (k === 'vector') {
		return `${name}, pass ${ph.iter + 1}: each warp reads its row with neighbouring lanes. ${reads.length} of ${N * WARP} lanes have an entry. The col reads take ${sectors}, one request per warp.`;
	}
	const busy = reads.filter((r) => !r.pad).length;
	const pads = reads.length - busy;
	const head = `${name}, iteration ${ph.iter + 1} of ${maxLen(p)}: `;
	if (k === 'ell') {
		return `${head}all ${N} threads read slot ${ph.iter} of their row, addresses ${ph.iter * N} to ${ph.iter * N + N - 1}, contiguous. ${busy} hold a nonzero, ${pads} padding. The col read takes ${sectors}.`;
	}
	const done = N - busy;
	return `${head}${busy} threads read entry ${ph.iter} of their row${done ? `, ${done} ${done === 1 ? 'has' : 'have'} finished and ${done === 1 ? 'waits' : 'wait'}` : ''}. The addresses are row_ptr[r] + ${ph.iter}, spread over the array: ${sectors} for the col read.`;
}

/** Code panel source per kernel (1-based lines used by lineAt). */
export const CODE: Record<KernelId, string> = {
	scalar: `int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows) {
    float sum = 0.0f;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i)
        sum += val[i] * x[col[i]];
    y[row] = sum;
}`,
	vector: `int lane = threadIdx.x % 32;
int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
if (row >= rows) return;
float sum = 0.0f;
for (int i = row_ptr[row] + lane; i < row_ptr[row + 1]; i += 32)
    sum += val[i] * x[col[i]];
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
if (lane == 0) y[row] = sum;`,
	ell: `int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows) {
    float sum = 0.0f;
    for (int k = 0; k < width; ++k) {
        int i = k * rows + row;
        sum += val[i] * x[col[i]];
    }
    y[row] = sum;
}`,
};

export function lineAt(k: KernelId, p: PresetId, step: number): number | null {
	const ph = phaseAt(k, p, step);
	if (ph.kind === 'start') return null;
	if (k === 'scalar') return ph.kind === 'write' ? 6 : 5;
	if (k === 'ell') return ph.kind === 'write' ? 8 : 6;
	return ph.kind === 'write' ? 9 : ph.kind === 'shuffle' ? 8 : 6;
}
