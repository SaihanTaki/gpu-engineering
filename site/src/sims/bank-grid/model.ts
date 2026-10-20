/**
 * One warp reads one 4-byte word per lane from a shared memory tile.
 *
 * CUDA Programming Guide, "Shared Memory Access Patterns": "Shared memory has
 * 32 banks that are organized such that successive 32-bit words map to
 * successive banks. Each bank has a bandwidth of 32 bits per clock cycle."
 * Reads of the same 32-bit word are broadcast, so they do not conflict.
 * https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns
 *
 * Each bank serves one word per wavefront, so the warp needs as many
 * wavefronts as the largest number of different words asked of one bank.
 */
export const BANKS = 32;
export const BANK_BYTES = 4;
export const WARP = 32;
/** Tile row width in floats, before padding. */
export const TILE = 32;
/** Enough rows for the largest stride: lane 31 at stride 64 reads row 62. */
export const TILE_ROWS = 64;

export type Layout = 'plain' | 'pad' | 'swizzle';
export const LAYOUTS: Record<Layout, { label: string; decl: string; read: string }> = {
	plain: { label: 'Plain', decl: `__shared__ float tile[${TILE_ROWS}][32];`, read: 'float v = tile[r][c];' },
	pad: { label: '+1 padding', decl: `__shared__ float tile[${TILE_ROWS}][32 + 1];`, read: 'float v = tile[r][c];' },
	swizzle: { label: 'XOR swizzle', decl: `__shared__ float tile[${TILE_ROWS}][32];`, read: 'float v = tile[r][c ^ (r % 32)];' },
};
export const LAYOUT_IDS = Object.keys(LAYOUTS) as Layout[];

export const STRIDES = [0, 1, 2, 3, 4, 8, 16, 17, 32, 33, 64];
export const SHARES = [1, 2, 4, 8];

export interface Config {
	/** Elements between the words of neighboring lane groups. 0 makes every lane read element 0. */
	stride: number;
	/** Neighboring lanes that read the same element. */
	share: number;
	layout: Layout;
}

export const codeFor = (layout: Layout) => `${LAYOUTS[layout].decl}
int lane = threadIdx.x % 32;
int k = (lane / G) * S;   // element this lane reads
int r = k / 32, c = k % 32;
${LAYOUTS[layout].read}`;
export const INDEX_LINES = [3, 4];
export const READ_LINE = 5;

export interface Access {
	lane: number;
	/** Logical element index, row r and column c of the 32-wide tile. */
	k: number;
	row: number;
	col: number;
	/** Physical 4-byte word offset from the start of the tile. */
	word: number;
	bank: number;
}

export function wordOf(layout: Layout, row: number, col: number) {
	if (layout === 'pad') return row * (TILE + 1) + col;
	if (layout === 'swizzle') return row * TILE + (col ^ (row % TILE));
	return row * TILE + col;
}

export function accesses(c: Config): Access[] {
	return Array.from({ length: WARP }, (_, lane) => {
		const k = Math.floor(lane / c.share) * c.stride;
		const row = Math.floor(k / TILE);
		const col = k % TILE;
		const word = wordOf(c.layout, row, col);
		return { lane, k, row, col, word, bank: word % BANKS };
	});
}

/** One word sent by one bank in one wavefront, with every lane that asked for it. */
export interface Serve {
	bank: number;
	/** 0-based wavefront index. */
	wave: number;
	word: number;
	lanes: number[];
}

/**
 * Wavefront w serves the w-th distinct word of every bank, in order of the
 * lowest lane asking for it. The hardware's order is not documented; only
 * the count matters.
 */
export function serves(c: Config): Serve[] {
	const perBank: Serve[][] = Array.from({ length: BANKS }, () => []);
	for (const a of accesses(c)) {
		const list = perBank[a.bank];
		const hit = list.find((s) => s.word === a.word);
		if (hit) hit.lanes.push(a.lane);
		else list.push({ bank: a.bank, wave: list.length, word: a.word, lanes: [a.lane] });
	}
	return perBank.flat();
}

/** Distinct words asked of each bank. */
export function wordsPerBank(c: Config) {
	const n = Array<number>(BANKS).fill(0);
	for (const s of serves(c)) n[s.bank]++;
	return n;
}

/** The conflict degree: the most distinct words any one bank must serve. Equal to the wavefront count. */
export const degree = (c: Config) => Math.max(...wordsPerBank(c));
export const lastStep = degree;

export interface State {
	step: number;
	/** Serves done so far, including the current wavefront. */
	done: Serve[];
	/** Serves of the wavefront shown at this step; empty at step 0. */
	current: Serve[];
}

export function stateAt(c: Config, step: number): State {
	const s = Math.max(0, Math.min(step, lastStep(c)));
	const all = serves(c);
	return { step: s, done: all.filter((x) => x.wave < s), current: all.filter((x) => x.wave === s - 1) };
}

/** "0 to 31", "0, 8, 16 and 24" or "3". */
export function listNumbers(ns: number[]): string {
	const xs = [...ns].sort((a, b) => a - b);
	if (xs.length > 2 && xs.every((x, i) => i === 0 || x === xs[i - 1] + 1)) return `${xs[0]} to ${xs[xs.length - 1]}`;
	if (xs.length <= 1) return xs.join('');
	if (xs.length > 8) return `${xs.slice(0, 6).join(', ')} and ${xs.length - 6} more`;
	return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	const d = degree(c);
	if (st.step === 0) {
		const a = accesses(c);
		const ex = [0, 1, 2]
			.map((i) => a[i])
			.map((x) => `lane ${x.lane} reads tile[${x.row}][${x.col}], word ${x.word}, bank ${x.bank}`)
			.join('; ');
		return `Every lane reads one 4-byte word: ${ex}. Bank = word % 32. Predict how many wavefronts the warp needs, then press Next.`;
	}
	const w = st.step;
	const banks = st.current.map((s) => s.bank);
	const lanes = st.current.reduce((n, s) => n + s.lanes.length, 0);
	const laneIds = st.current.flatMap((s) => s.lanes);
	const who = lanes === 1 ? `lane ${laneIds[0]}` : `${lanes} lanes`;
	let text =
		banks.length === 1
			? `Wavefront ${w}: bank ${banks[0]} sends one word to ${who}.`
			: `Wavefront ${w}: ${banks.length} banks (${listNumbers(banks)}) send one word each, to ${who}.`;
	if (w > 1) {
		text += ` ${banks.length === 1 ? 'That bank' : 'Each of these banks'} already sent a different word in an earlier wavefront, so ${lanes === 1 ? 'this lane' : 'these lanes'} had to wait. The wait is the bank conflict.`;
	}
	const shared = st.current.filter((s) => s.lanes.length > 1);
	if (shared.length) {
		const s = shared[0];
		text +=
			shared.length === 1
				? ` Bank ${s.bank} broadcasts word ${s.word} to lanes ${listNumbers(s.lanes)} at once.`
				: ` ${shared.length} banks broadcast one word each to several lanes, for example bank ${s.bank} to lanes ${listNumbers(s.lanes)}.`;
	}
	if (w === d) {
		text +=
			d === 1
				? ' Done: one wavefront served all 32 lanes, so the read is conflict-free.'
				: ` Done: ${d} wavefronts instead of 1, a ${d}-way bank conflict.`;
	}
	return text;
}
