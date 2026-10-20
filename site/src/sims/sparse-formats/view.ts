import { clear, svg } from '@sim-core/svg';
import {
	csr,
	ell,
	FORMAT_NAMES,
	lengths,
	maxLen,
	N,
	pattern,
	phaseAt,
	PER_SECTOR,
	readsAt,
	value,
	vectorPasses,
	WARP,
	type FormatId,
	type KernelId,
	type PresetId,
} from './model';

export interface ViewConfig {
	preset: PresetId;
	kernel: KernelId;
	format: FormatId;
}

const PAD = 12;
const ROW_LABEL = 26;
const STRIP_LABEL = 60;
const GAP = 28;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/** Read state of every CSR entry and every ELL slot at this step. */
function readState(c: ViewConfig, step: number) {
	const now = new Set<number>();
	const done = new Set<number>();
	const slotNow = new Set<string>();
	const slotDone = new Set<string>();
	for (let s = 1; s <= step; s++) {
		for (const r of readsAt(c.kernel, c.preset, s)) {
			const key = `${r.row}:${r.slot}`;
			if (s === step) {
				if (r.csrIndex !== null) now.add(r.csrIndex);
				slotNow.add(key);
			} else {
				if (r.csrIndex !== null) done.add(r.csrIndex);
				slotDone.add(key);
			}
		}
	}
	return { now, done, slotNow, slotDone };
}

const state = (now: boolean, done: boolean) => (now ? ' is-now' : done ? ' is-done' : '');

function cell(nodes: SVGElement[], x: number, y: number, size: number, cls: string, label: string | null) {
	nodes.push(rect(x, y, size, size, cls));
	if (label !== null && size >= 15) nodes.push(text(x + size / 2, y + size / 2 + 4, label, `sim-cell-text sp-cell-text${cls.includes('is-now') ? ' is-now' : ''}`));
}

/** Work per thread (scalar, ELL) or lanes per warp (vector), one bar per matrix row. */
function drawBars(nodes: SVGElement[], c: ViewConfig, step: number, x0: number, y0: number, w: number, rowH: number) {
	const lens = lengths(c.preset);
	const ph = phaseAt(c.kernel, c.preset, step);
	const vector = c.kernel === 'vector';
	const units = vector ? WARP * vectorPasses(c.preset) : maxLen(c.preset);
	const label = vector ? 'Lanes of each row’s warp' : 'Loop iterations of each row’s thread';
	nodes.push(text(x0, y0 - 8, label, 'sim-small'));
	const barX = x0 + ROW_LABEL;
	const u = (w - ROW_LABEL) / units;
	const h = Math.min(rowH - 4, 16);
	const doneIters = ph.kind === 'loop' ? ph.iter : ph.kind === 'start' ? 0 : units;
	for (let r = 0; r < N; r++) {
		const y = y0 + r * rowH + (rowH - h) / 2;
		nodes.push(text(x0, y + h / 2 + 4, `${vector ? 'w' : 't'}${r}`, 'sim-small'));
		for (let j = 0; j < units; j++) {
			const work = j < lens[r];
			const base = work ? 'sp-work' : c.kernel === 'ell' ? 'sp-pad' : 'sp-idle';
			let now: boolean;
			let done: boolean;
			if (vector) {
				// Lane j of pass p is unit p * 32 + j.
				const pass = Math.floor(j / WARP);
				now = ph.kind === 'loop' && pass === ph.iter;
				done = (ph.kind === 'loop' && pass < ph.iter) || ph.kind === 'shuffle' || ph.kind === 'write';
			} else {
				now = ph.kind === 'loop' && j === ph.iter;
				done = j < doneIters;
			}
			nodes.push(rect(barX + j * u + 0.5, y, Math.max(1, u - 1), h, base + state(now, done)));
		}
	}
	return y0 + N * rowH;
}

export function render(root: SVGSVGElement, c: ViewConfig, step: number, width: number) {
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	const rows = pattern(c.preset);
	const a = csr(c.preset);
	const e = ell(c.preset);
	const rs = readState(c, step);
	let y = PAD;

	// Matrix, with the work bars to its right when there is room, else below it.
	const cs = Math.max(14, Math.min(24, Math.floor((inner - ROW_LABEL) / N)));
	const matW = ROW_LABEL + N * cs;
	const side = inner - matW - GAP >= 200;
	nodes.push(text(PAD, y + 12, `A, ${N} x ${N}`));
	y += 34;
	const top = y;
	for (let r = 0; r < N; r++) {
		nodes.push(text(PAD, top + r * cs + cs / 2 + 4, `r${r}`, 'sim-small'));
		for (let col = 0; col < N; col++) {
			const x = PAD + ROW_LABEL + col * cs;
			const k = rows[r].indexOf(col);
			if (k < 0) {
				nodes.push(rect(x, top + r * cs, cs, cs, 'sp-empty'));
				continue;
			}
			const i = a.rowPtr[r] + k;
			cell(nodes, x, top + r * cs, cs, `sp-nz${state(rs.now.has(i), rs.done.has(i))}`, String(value(r, col)));
		}
	}
	y = top + N * cs;
	if (side) {
		drawBars(nodes, c, step, PAD + matW + GAP, top, inner - matW - GAP, cs);
	} else {
		y = drawBars(nodes, c, step, PAD, y + 30, inner, Math.max(14, Math.min(cs, 18)));
	}
	y += 22;

	// Arrays of the chosen format.
	const kernelReadsHere = (c.format === 'ell') === (c.kernel === 'ell');
	nodes.push(text(PAD, y + 12, `${FORMAT_NAMES[c.format]} arrays`));
	nodes.push(
		text(PAD, y + 28, subtitle(c.format, kernelReadsHere), 'sim-small'),
	);
	y += 42;

	if (c.format === 'ell') {
		y = drawEll(nodes, e, rs, y, inner);
	} else {
		if (c.format === 'csr') y = drawStrips(nodes, [{ name: 'row_ptr', values: a.rowPtr }], null, y, inner, false);
		const strips = [
			...(c.format === 'coo' ? [{ name: 'row', values: a.row }] : []),
			{ name: 'col', values: a.col },
			{ name: 'val', values: a.val },
		];
		y = drawStrips(nodes, strips, rs, y, inner, kernelReadsHere);
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y + PAD}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y + PAD));
	root.append(...nodes);
}

function subtitle(f: FormatId, kernelReadsHere: boolean) {
	if (f === 'ell') return 'rows down, slots across, stored by slot';
	return kernelReadsHere ? 'ticks mark 32-byte sectors' : 'the same entries in CSR order';
}

interface Strip {
	name: string;
	values: number[];
}

/** Arrays drawn as rows of cells, wrapped; strips of equal length share each wrapped line. */
function drawStrips(nodes: SVGElement[], strips: Strip[], rs: ReturnType<typeof readState> | null, y: number, inner: number, ticks: boolean) {
	const n = strips[0].values.length;
	const cw = 20;
	// Whole sectors per line, so the ticks line up from one line to the next.
	const perLine = Math.max(PER_SECTOR, Math.floor((inner - STRIP_LABEL) / cw / PER_SECTOR) * PER_SECTOR);
	for (let start = 0; start < n; start += perLine) {
		const end = Math.min(n, start + perLine);
		nodes.push(text(PAD + STRIP_LABEL, y + 9, String(start), 'sim-small'));
		y += 14;
		const lineTop = y;
		for (const s of strips) {
			nodes.push(text(PAD, y + cw / 2 + 4, s.name, 'sim-small'));
			for (let i = start; i < end; i++) {
				const cls = rs ? `sp-nz${state(rs.now.has(i), rs.done.has(i))}` : 'sp-ptr';
				cell(nodes, PAD + STRIP_LABEL + (i - start) * cw, y, cw, cls, String(s.values[i]));
			}
			y += cw + 2;
		}
		if (ticks) {
			for (let i = start; i <= end; i += PER_SECTOR) {
				if (i % PER_SECTOR) continue;
				const x = PAD + STRIP_LABEL + (i - start) * cw;
				nodes.push(svg('line', { x1: x, x2: x, y1: lineTop - 4, y2: y + 2, class: 'sp-tick' }));
			}
		}
		y += 8;
	}
	return y;
}

/** ELL col and val as rows x slots grids; memory order runs down each slot column. */
function drawEll(nodes: SVGElement[], e: ReturnType<typeof ell>, rs: ReturnType<typeof readState>, y: number, inner: number) {
	const sideBySide = (inner - ROW_LABEL * 2 - GAP) / (2 * e.width) >= 16;
	const ce = Math.min(22, Math.floor(sideBySide ? (inner - ROW_LABEL * 2 - GAP) / (2 * e.width) : (inner - ROW_LABEL) / e.width));
	const gridW = ROW_LABEL + e.width * ce;
	const grids = [
		{ name: 'col', values: e.col },
		{ name: 'val', values: e.val },
	];
	let bottom = y;
	grids.forEach((g, gi) => {
		const x0 = PAD + (sideBySide ? gi * (gridW + GAP) : 0);
		const y0 = sideBySide ? y : y + gi * (N * ce + 30);
		nodes.push(text(x0, y0 + 9, g.name, 'sim-small'));
		const top = y0 + 16;
		for (let r = 0; r < N; r++) {
			nodes.push(text(x0, top + r * ce + ce / 2 + 4, `r${r}`, 'sim-small'));
			for (let k = 0; k < e.width; k++) {
				const i = k * N + r;
				const key = `${r}:${k}`;
				const base = e.pad[i] ? 'sp-pad' : 'sp-nz';
				cell(nodes, x0 + ROW_LABEL + k * ce, top + r * ce, ce, base + state(rs.slotNow.has(key), rs.slotDone.has(key)), String(g.values[i]));
			}
		}
		bottom = Math.max(bottom, top + N * ce);
	});
	return bottom + 8;
}
