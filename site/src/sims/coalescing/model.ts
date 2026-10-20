/**
 * One warp's load instruction, mapped onto memory. Every lane reads one
 * element; the model counts the 32-byte sectors and 128-byte lines the warp
 * touches.
 *
 * Sources for the constants:
 * - CUDA Programming Guide, "Coalesced Global Memory Access": "Global memory is
 *   accessed via 32-byte memory transactions."
 * - CUDA C++ Best Practices Guide, "Coalesced Access to Global Memory": on compute
 *   capability 6.0 and higher a warp's accesses "coalesce into a number of
 *   transactions equal to the number of 32-byte transactions necessary to service
 *   all of the threads of the warp".
 * - Nsight Compute Profiling Guide, "Quantities": "An L1 or L2 cache line is four
 *   sectors, i.e. 128 bytes."
 * - Best Practices Guide: cudaMalloc memory "is guaranteed to be aligned to at
 *   least 256 bytes", so every array here starts at address 0.
 */
export const WARP = 32;
export const SECTOR = 32;
export const LINE = 128;

export type Pattern = 'strided' | 'aos' | 'soa';
export const PATTERNS: Record<Pattern, string> = {
	strided: 'Array: in[i * STRIDE + OFFSET]',
	aos: 'Array of structs: p[i].f0',
	soa: 'Struct of arrays: p.f0[i]',
};
export const PATTERN_IDS = Object.keys(PATTERNS) as Pattern[];

export const ELEM_BYTES = [1, 2, 4, 8, 16] as const;
export type ElemBytes = (typeof ELEM_BYTES)[number];
export const TYPE_NAMES: Record<ElemBytes, string> = { 1: 'char', 2: 'short', 4: 'float', 8: 'double', 16: 'float4' };

export interface Config {
	pattern: Pattern;
	elemBytes: ElemBytes;
	/** Strided pattern only: elements between neighbouring lanes, and the start offset in elements. */
	stride: number;
	offset: number;
	/** AoS and SoA: fields per struct, each of elemBytes. */
	fields: number;
}

export interface Access {
	lane: number;
	/** Element index into the array (AoS: into the array viewed as plain elements). */
	index: number;
	/** Byte address relative to the start of the allocation. */
	addr: number;
	sector: number;
	line: number;
}

/** Elements between lane k and lane k+1, and the element lane 0 reads. */
export function layout(c: Config) {
	if (c.pattern === 'aos') return { stride: c.fields, offset: 0 };
	if (c.pattern === 'soa') return { stride: 1, offset: 0 };
	return { stride: c.stride, offset: c.offset };
}

export function access(c: Config, lane: number): Access {
	const { stride, offset } = layout(c);
	const index = lane * stride + offset;
	const addr = index * c.elemBytes;
	// Elements are naturally aligned and at most 16 bytes, so one never spans two sectors.
	return { lane, index, addr, sector: Math.floor(addr / SECTOR), line: Math.floor(addr / LINE) };
}

export interface Analysis {
	accesses: Access[];
	/** Sectors and lines touched, ascending. */
	sectors: number[];
	lines: number[];
	/** Distinct bytes the lanes ask for, and bytes the sectors bring in. */
	usedBytes: number;
	fetchedBytes: number;
	efficiency: number;
}

const uniq = (xs: number[]) => [...new Set(xs)].sort((a, b) => a - b);

export function analyze(c: Config, lanes = WARP): Analysis {
	const accesses = Array.from({ length: lanes }, (_, lane) => access(c, lane));
	const sectors = uniq(accesses.map((a) => a.sector));
	const usedBytes = uniq(accesses.map((a) => a.addr)).length * c.elemBytes;
	const fetchedBytes = sectors.length * SECTOR;
	return {
		accesses,
		sectors,
		lines: uniq(accesses.map((a) => a.line)),
		usedBytes,
		fetchedBytes,
		efficiency: fetchedBytes ? usedBytes / fetchedBytes : 0,
	};
}

/** Step 0 is before the load; step k has placed lanes 0 to k-1. */
export const lastStep = () => WARP;

export interface State {
	step: number;
	/** The lane placed by this step, or null at step 0. */
	current: Access | null;
	/** True when the current lane's sector was already needed by an earlier lane. */
	reused: boolean;
	sofar: Analysis;
	/** The whole warp, so the drawing can lay out every line from the start. */
	full: Analysis;
}

export function stateAt(c: Config, step: number): State {
	const s = Math.max(0, Math.min(step, WARP));
	const sofar = analyze(c, s);
	const current = s ? sofar.accesses[s - 1] : null;
	const reused = current ? sofar.accesses.slice(0, s - 1).some((a) => a.sector === current.sector) : false;
	return { step: s, current, reused, sofar, full: analyze(c) };
}

export const pct = (f: number) => `${+(100 * f).toFixed(1)}%`;
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

export function summary(a: Analysis) {
	return `${plural(a.sectors.length, 'sector')} in ${plural(a.lines.length, 'line')}: ${a.fetchedBytes} bytes fetched for ${a.usedBytes} bytes used, ${pct(a.efficiency)} efficiency.`;
}

export function patternText(c: Config) {
	const t = TYPE_NAMES[c.elemBytes];
	if (c.pattern === 'aos') return `Each lane reads field f0 of its own struct of ${c.fields} ${t} fields (${c.fields * c.elemBytes} bytes per struct).`;
	if (c.pattern === 'soa') return `Each lane reads element i of the f0 array; the ${c.fields} fields live in ${c.fields} separate arrays.`;
	return `Each lane reads in[i * ${c.stride} + ${c.offset}], a ${c.elemBytes}-byte ${t}.`;
}

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	if (!st.current) {
		return `${patternText(c)} Predict: how many 32-byte sectors does the warp need? Then press Next to place the lanes one by one.`;
	}
	const a = st.current;
	let text = `Lane ${a.lane} reads bytes ${a.addr} to ${a.addr + c.elemBytes - 1}, in sector ${a.sector} of line ${a.line}. `;
	if (st.reused) text += `An earlier lane already needs that sector, so no new transaction. The warp needs ${plural(st.sofar.sectors.length, 'sector')} so far.`;
	else text += `That sector is new: the warp now needs ${plural(st.sofar.sectors.length, 'sector')}.`;
	if (st.step === WARP) text += ` Done: ${summary(st.full)}`;
	return text;
}

/** Source shown in the code panel for each pattern. Line 3 is the load the sim steps through. */
export const LOAD_LINE = 3;
export function codeFor(p: Pattern) {
	if (p === 'aos') {
		return `// struct S { T f0, f1, ...; };   S *p;
int i = blockIdx.x * blockDim.x + threadIdx.x;
T v = p[i].f0;
out[i] = v;`;
	}
	if (p === 'soa') {
		return `// struct SoA { T *f0, *f1, ...; }; SoA p;
int i = blockIdx.x * blockDim.x + threadIdx.x;
T v = p.f0[i];
out[i] = v;`;
	}
	return `// T *in, *out;
int i = blockIdx.x * blockDim.x + threadIdx.x;
T v = in[i * STRIDE + OFFSET];
out[i] = v;`;
}
