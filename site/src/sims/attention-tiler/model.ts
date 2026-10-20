/**
 * FlashAttention forward pass, FlashAttention-2 loop order (Dao 2023, arXiv 2307.08691,
 * Algorithm 1): the outer loop runs over blocks of Br rows of Q, the inner loop over blocks
 * of Bc rows of K and V. On a GPU every outer iteration is its own program; here they run
 * one after another so each tile can be stepped.
 *
 * The values are real: Q, K and V hold small numbers, and the running row max m, row sum l
 * and output accumulator are computed with the online softmax, so the final O equals
 * softmax(Q K^T / sqrt(d)) V (checked in model.test.ts).
 *
 * HBM traffic counts elements moved between HBM and the chip, ELEM_BYTES each (FP16):
 *  - flash: Q_i once per Q block, K_j and V_j once per visited tile, O_i once per Q block.
 *  - naive, Algorithm 0 of Dao et al. 2022 (arXiv 2205.14135): S = QK^T reads Q, K and
 *    writes S; P = softmax(S) reads S, writes P; O = PV reads P, V and writes O. Each matmul
 *    is assumed to read its inputs exactly once, the best a tiled matmul can do.
 * Neither count includes L2 cache hits, which in practice serve many of the K/V re-reads.
 */
export const ELEM_BYTES = 2;
export const ACC_BYTES = 4;

export const N_OPTIONS = [8, 12, 16] as const;
export const D_OPTIONS = [4, 8] as const;
export const BLOCK_OPTIONS = [2, 4, 8] as const;
export const MAX_N = 16;
export const MAX_D = 8;

export interface Config {
	n: number;
	d: number;
	br: number;
	bc: number;
	causal: boolean;
}

export const DEFAULT_CONFIG: Config = { n: 12, d: 4, br: 4, bc: 4, causal: false };
/** A realistic single-head size for the same formulas: FlashAttention-2 picks blocks from {64, 128}. */
export const REAL_CONFIG: Config = { n: 4096, d: 64, br: 128, bc: 64, causal: false };

type Matrix = number[][];

/** Deterministic values in [-2, 2] with one decimal, the same for every n and d (sliced). */
function table(seed: number): Matrix {
	let s = seed >>> 0;
	const rand = () => {
		// mulberry32
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
	return Array.from({ length: MAX_N }, () => Array.from({ length: MAX_D }, () => Math.round(rand() * 40 - 20) / 10));
}
const TABLES = { q: table(1), k: table(2), v: table(3) };

export function inputs(n: number, d: number) {
	const cut = (m: Matrix) => m.slice(0, n).map((r) => r.slice(0, d));
	return { q: cut(TABLES.q), k: cut(TABLES.k), v: cut(TABLES.v) };
}

/** Standard attention: materialize each row of S, softmax it, multiply by V. */
export function referenceAttention(q: Matrix, k: Matrix, v: Matrix, causal: boolean): Matrix {
	const d = q[0].length;
	const scale = 1 / Math.sqrt(d);
	return q.map((qr, i) => {
		const cols = causal ? i + 1 : k.length;
		const s = k.slice(0, cols).map((kr) => qr.reduce((a, x, c) => a + x * kr[c], 0) * scale);
		const mx = Math.max(...s);
		const p = s.map((x) => Math.exp(x - mx));
		const sum = p.reduce((a, x) => a + x, 0);
		return Array.from({ length: d }, (_, c) => p.reduce((a, x, j) => a + x * v[j][c], 0) / sum);
	});
}

export const blocks = (n: number, b: number) => Math.ceil(n / b);
const span = (i: number, b: number, n: number) => ({ start: i * b, end: Math.min(n, (i + 1) * b) });

/** A causal tile is skipped when its first column is past the Q block's last row. */
export function tileSkipped(c: Config, i: number, j: number) {
	return c.causal && j * c.bc > span(i, c.br, c.n).end - 1;
}
/** A visited causal tile needs the mask when some column is past some row. */
export function tileMasked(c: Config, i: number, j: number) {
	return c.causal && span(j, c.bc, c.n).end - 1 > i * c.br;
}

export type Step =
	| { kind: 'start' }
	| { kind: 'tile'; i: number; j: number; first: boolean; masked: boolean }
	| { kind: 'write'; i: number };

export function steps(c: Config): Step[] {
	const out: Step[] = [{ kind: 'start' }];
	for (let i = 0; i < blocks(c.n, c.br); i++) {
		let first = true;
		for (let j = 0; j < blocks(c.n, c.bc); j++) {
			if (tileSkipped(c, i, j)) continue;
			out.push({ kind: 'tile', i, j, first, masked: tileMasked(c, i, j) });
			first = false;
		}
		out.push({ kind: 'write', i });
	}
	return out;
}

export interface Traffic {
	read: number;
	written: number;
}

/** HBM bytes one step moves. */
export function stepTraffic(c: Config, s: Step): Traffic {
	if (s.kind === 'start') return { read: 0, written: 0 };
	if (s.kind === 'write') {
		const r = span(s.i, c.br, c.n);
		return { read: 0, written: (r.end - r.start) * c.d * ELEM_BYTES };
	}
	const cols = span(s.j, c.bc, c.n);
	const rows = span(s.i, c.br, c.n);
	const q = s.first ? (rows.end - rows.start) * c.d : 0;
	return { read: (q + 2 * (cols.end - cols.start) * c.d) * ELEM_BYTES, written: 0 };
}

/** Whole forward pass, by the same per-step rule (works for large n without building values). */
export function flashTraffic(c: Config): Traffic & { tiles: number; skipped: number } {
	let read = 0;
	let written = 0;
	let tiles = 0;
	let skipped = 0;
	for (let i = 0; i < blocks(c.n, c.br); i++) {
		let first = true;
		for (let j = 0; j < blocks(c.n, c.bc); j++) {
			if (tileSkipped(c, i, j)) {
				skipped++;
				continue;
			}
			read += stepTraffic(c, { kind: 'tile', i, j, first, masked: false }).read;
			first = false;
			tiles++;
		}
		written += stepTraffic(c, { kind: 'write', i }).written;
	}
	return { read, written, tiles, skipped };
}

/** Algorithm 0: reads 3nd + 2n^2 elements, writes 2n^2 + nd. The mask does not shrink S or P. */
export function naiveTraffic(c: Config): Traffic & { scoreBytes: number } {
	const nd = c.n * c.d;
	const nn = c.n * c.n;
	return { read: (3 * nd + 2 * nn) * ELEM_BYTES, written: (2 * nn + nd) * ELEM_BYTES, scoreBytes: 2 * nn * ELEM_BYTES };
}

/** What one Q block keeps on chip while it runs: FP16 tiles, FP32 scores, accumulator and m, l. */
export function onChip(c: Config) {
	const tiles = (c.br * c.d + 2 * c.bc * c.d) * ELEM_BYTES;
	const scores = c.br * c.bc * ACC_BYTES;
	const acc = c.br * c.d * ACC_BYTES;
	const stats = 2 * c.br * ACC_BYTES;
	return { tiles, scores, acc, stats, total: tiles + scores + acc + stats };
}

/** Forward FLOPs as FlashAttention-2 counts them: 4 n^2 d, halved with the causal mask. */
export const attentionFlops = (c: Config) => (4 * c.n * c.n * c.d) / (c.causal ? 2 : 1);

export interface RowStat {
	row: number;
	mOld: number;
	mNew: number;
	lOld: number;
	lNew: number;
	/** exp(m_old - m_new): what the earlier sum and accumulator are multiplied by. */
	alpha: number;
	accOld: number;
	accNew: number;
	/** Set on the write step: acc / l for column 0. */
	out?: number;
}

export interface Frame {
	step: Step;
	/** Cumulative HBM bytes up to and including this step. */
	read: number;
	written: number;
	rows: RowStat[];
}

export interface Run {
	frames: Frame[];
	/** The output O after the last step. */
	out: Matrix;
	/** Step index at which each tile runs, -1 if skipped. Indexed [i][j]. */
	tileStep: number[][];
	/** Step index at which each Q block's O rows are written. */
	writeStep: number[];
}

export function simulate(c: Config): Run {
	const { q, k, v } = inputs(c.n, c.d);
	const scale = 1 / Math.sqrt(c.d);
	const list = steps(c);
	const m = Array<number>(c.n).fill(-Infinity);
	const l = Array<number>(c.n).fill(0);
	const acc: Matrix = Array.from({ length: c.n }, () => Array<number>(c.d).fill(0));
	const out: Matrix = Array.from({ length: c.n }, () => Array<number>(c.d).fill(0));
	const tileStep = Array.from({ length: blocks(c.n, c.br) }, () => Array<number>(blocks(c.n, c.bc)).fill(-1));
	const writeStep = Array<number>(blocks(c.n, c.br)).fill(-1);
	const frames: Frame[] = [];
	let read = 0;
	let written = 0;

	list.forEach((s, idx) => {
		const t = stepTraffic(c, s);
		read += t.read;
		written += t.written;
		const rows: RowStat[] = [];
		if (s.kind === 'tile') {
			tileStep[s.i][s.j] = idx;
			const R = span(s.i, c.br, c.n);
			const C = span(s.j, c.bc, c.n);
			for (let r = R.start; r < R.end; r++) {
				const sc: number[] = [];
				for (let j = C.start; j < C.end; j++) {
					sc.push(c.causal && j > r ? -Infinity : q[r].reduce((a, x, e) => a + x * k[j][e], 0) * scale);
				}
				const mNew = Math.max(m[r], ...sc);
				const alpha = Math.exp(m[r] - mNew);
				const p = sc.map((x) => Math.exp(x - mNew));
				const stat: RowStat = { row: r, mOld: m[r], mNew, lOld: l[r], lNew: 0, alpha, accOld: acc[r][0], accNew: 0 };
				l[r] = l[r] * alpha + p.reduce((a, x) => a + x, 0);
				for (let e = 0; e < c.d; e++) acc[r][e] = acc[r][e] * alpha + p.reduce((a, x, jj) => a + x * v[C.start + jj][e], 0);
				m[r] = mNew;
				stat.lNew = l[r];
				stat.accNew = acc[r][0];
				rows.push(stat);
			}
		} else if (s.kind === 'write') {
			writeStep[s.i] = idx;
			const R = span(s.i, c.br, c.n);
			for (let r = R.start; r < R.end; r++) {
				for (let e = 0; e < c.d; e++) out[r][e] = acc[r][e] / l[r];
				rows.push({ row: r, mOld: m[r], mNew: m[r], lOld: l[r], lNew: l[r], alpha: 1, accOld: acc[r][0], accNew: acc[r][0], out: out[r][0] });
			}
		}
		frames.push({ step: s, read, written, rows });
	});
	return { frames, out, tileStep, writeStep };
}

export type TileState = 'pending' | 'current' | 'done' | 'skipped';

export function tileState(run: Run, i: number, j: number, step: number): TileState {
	const at = run.tileStep[i][j];
	if (at < 0) return 'skipped';
	if (at === step) return 'current';
	return at < step ? 'done' : 'pending';
}

export function fmtBytes(b: number) {
	const units = ['bytes', 'KiB', 'MiB', 'GiB'];
	let i = 0;
	while (b >= 1024 && i < units.length - 1) {
		b /= 1024;
		i++;
	}
	return `${i === 0 ? fmt(b) : +b.toFixed(2)} ${units[i]}`;
}
export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
/** Two decimals; -inf as text. */
export const num = (x: number) => (x === -Infinity ? '-inf' : (Math.abs(x) < 0.005 ? 0 : x).toFixed(2));

const rowsText = (s: { start: number; end: number }) => (s.end - s.start === 1 ? `row ${s.start}` : `rows ${s.start} to ${s.end - 1}`);

export function describe(c: Config, run: Run, step: number): string {
	const f = run.frames[step];
	const s = f.step;
	const naive = naiveTraffic(c);
	const tr = blocks(c.n, c.br);
	const tc = blocks(c.n, c.bc);
	if (s.kind === 'start') {
		const skip = c.causal ? ` The causal mask lets the loop skip ${flashTraffic(c).skipped} of the ${tr * tc} tiles.` : '';
		return (
			`n = ${c.n}, d = ${c.d}: Q splits into ${tr} blocks of ${c.br} rows, K and V into ${tc} blocks of ${c.bc} rows. ` +
			`Naive attention moves ${fmtBytes(naive.read + naive.written)} through HBM, writing the ${c.n} x ${c.n} matrices S and P.${skip} ` +
			`Guess how many bytes the tiled version moves, then step.`
		);
	}
	if (s.kind === 'write') {
		const R = span(s.i, c.br, c.n);
		const done = step === run.frames.length - 1;
		const tail = done
			? ` Done: ${fmtBytes(f.read + f.written)} moved against ${fmtBytes(naive.read + naive.written)} for naive attention, and S never left the chip.`
			: '';
		return `Q block ${s.i} finished: O for ${rowsText(R)} is acc / l, written to HBM once (${fmtBytes(stepTraffic(c, s).written)}). m, l and acc are discarded.${tail}`;
	}
	const R = span(s.i, c.br, c.n);
	const C = span(s.j, c.bc, c.n);
	const t = stepTraffic(c, s);
	const loads = s.first ? `Q block ${s.i} (${rowsText(R)}) and K, V block ${s.j}` : `K and V block ${s.j}`;
	const mask = s.masked ? ' Entries above the diagonal are masked to -inf.' : '';
	// The row whose earlier work shrinks the most shows the rescale best.
	const r = f.rows.reduce((a, b) => (b.alpha < a.alpha ? b : a));
	const rescale =
		r.mOld === -Infinity
			? `First tile for these rows: m starts at the tile's row max, for example row ${r.row}: m = ${num(r.mNew)}, l = ${num(r.lNew)}.`
			: r.alpha < 1
				? `Row ${r.row}: max rises from ${num(r.mOld)} to ${num(r.mNew)}, so the old l and acc are multiplied by exp(${num(r.mOld)} - ${num(r.mNew)}) = ${num(r.alpha)}: l = ${num(r.lOld)} x ${num(r.alpha)} + ${num(r.lNew - r.lOld * r.alpha)} = ${num(r.lNew)}.`
				: `No row max rises in this tile, so nothing is rescaled; the new exponentials are added to l and acc.`;
	return `Tile S(${s.i}, ${s.j}): load ${loads} from HBM (${fmtBytes(t.read)}), compute the ${R.end - R.start} x ${C.end - C.start} score tile on chip.${mask} ${rescale}`;
}

/** Code panel: the Triton kernel's loop, shortened. */
export const CODE = `q = tl.load(q_ptrs)                       # Q_i, Br x d
m = tl.full([BLOCK_M], -inf)
l = tl.zeros([BLOCK_M])
acc = tl.zeros([BLOCK_M, HEAD_D])
for start_n in range(0, hi, BLOCK_N):     # K/V blocks
    k = tl.load(k_ptrs); v = tl.load(v_ptrs)
    s = tl.dot(q, tl.trans(k)) * sm_scale  # S_ij, Br x Bc
    s = tl.where(keep, s, -inf)            # causal mask
    m_new = tl.maximum(m, tl.max(s, 1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new[:, None])
    l = l * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p, v)
    m = m_new
tl.store(o_ptrs, acc / l[:, None])        # O_i`;

/** Highlighted lines (1-based) for a step. */
export function linesAt(s: Step): number[] {
	if (s.kind === 'start') return [];
	if (s.kind === 'write') return [15];
	const body = [5, 6, 7, ...(s.masked ? [8] : []), 9, 10, 11, 12, 13, 14];
	return s.first ? [1, 2, 3, 4, ...body] : body;
}
