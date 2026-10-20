import { clear, svg } from '@sim-core/svg';
import { blockOf, type Config, type State } from './model';

const PAD = 12;
const GAP = 16;
/** Room above a grid for its label. */
const LABEL = 20;

type Mark = 'plain' | 'strip' | 'read';

function grid(
	out: SVGElement[],
	x0: number,
	y0: number,
	rows: number,
	cols: number,
	cell: number,
	mark: (r: number, c: number) => Mark,
	label: (r: number, c: number) => string,
) {
	const showText = cell >= 15;
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < cols; c++) {
			const m = mark(r, c);
			const x = x0 + c * cell;
			const y = y0 + r * cell;
			out.push(svg('rect', { x, y, width: cell, height: cell, class: `tl-cell is-${m}` }));
			const t = label(r, c);
			if (showText && t) out.push(svg('text', { x: x + cell / 2, y: y + cell / 2 + 4, class: `sim-cell-text tl-count is-${m}` }, [t]));
		}
}

const outline = (x: number, y: number, w: number, h: number) => svg('rect', { x, y, width: w, height: h, class: 'tl-outline' });

export function render(root: SVGSVGElement, c: Config, st: State, width: number) {
	const inner = width - 2 * PAD;
	const n = c.n;
	const T = c.tile;
	const { bx, by } = blockOf(c);
	const shared = c.mode === 'shared';
	const x = st.cur;

	// Matrices: A on the left, B above C, so a row of A and a column of B meet at C.
	const cell = Math.max(6, Math.min(22, Math.floor((inner - GAP) / (2 * n))));
	const matW = 2 * n * cell + GAP;
	const scell = Math.max(14, Math.min(26, Math.floor((inner - GAP) / (2 * T))));
	const smemW = 2 * T * scell + GAP;
	const side = shared && matW + 2 * GAP + smemW <= inner;

	const ax = PAD;
	const cx = PAD + n * cell + GAP;
	const byTop = PAD + LABEL;
	const cy = byTop + n * cell + LABEL;
	const out: SVGElement[] = [];

	const inStripA = (r: number) => r >= by * T && r < (by + 1) * T;
	const inStripB = (q: number) => q >= bx * T && q < (bx + 1) * T;
	const readA = (r: number, q: number) =>
		inStripA(r) && ((x.kind === 'compute' && !shared && q === x.k) || (x.kind === 'load' && q >= x.phase * T && q < (x.phase + 1) * T));
	const readB = (r: number, q: number) =>
		inStripB(q) && ((x.kind === 'compute' && !shared && r === x.k) || (x.kind === 'load' && r >= x.phase * T && r < (x.phase + 1) * T));
	const count = (v: number) => (v ? String(v) : '');

	out.push(svg('text', { x: ax, y: cy - 6, class: 'sim-label' }, ['A']));
	grid(out, ax, cy, n, n, cell, (r, q) => (readA(r, q) ? 'read' : inStripA(r) ? 'strip' : 'plain'), (r, q) => count(st.aCount[r * n + q]));
	out.push(svg('text', { x: cx, y: byTop - 6, class: 'sim-label' }, ['B']));
	grid(out, cx, byTop, n, n, cell, (r, q) => (readB(r, q) ? 'read' : inStripB(q) ? 'strip' : 'plain'), (r, q) => count(st.bCount[r * n + q]));
	out.push(svg('text', { x: cx, y: cy - 6, class: 'sim-label' }, [`C, block (${bx}, ${by})`]));
	const inBlock = (r: number, q: number) => inStripA(r) && inStripB(q);
	grid(out, cx, cy, n, n, cell, (r, q) => (inBlock(r, q) ? (st.stored ? 'read' : 'strip') : 'plain'), (r, q) => (inBlock(r, q) && st.terms ? String(st.terms) : ''));
	out.push(outline(cx + bx * T * cell, cy + by * T * cell, T * cell, T * cell));
	// The tile of A and of B the current phase works on.
	if (shared && st.tileA && st.tileB) {
		out.push(outline(ax + st.tileA.c0 * cell, cy + st.tileA.r0 * cell, T * cell, T * cell));
		out.push(outline(cx + st.tileB.c0 * cell, byTop + st.tileB.r0 * cell, T * cell, T * cell));
	}
	let height = cy + n * cell;

	if (shared) {
		const sx = side ? PAD + matW + 2 * GAP : PAD;
		const sy = side ? cy : height + LABEL + 8;
		const loaded = st.tileA !== null;
		const compute = x.kind === 'compute';
		const mark = (on: boolean): Mark => (on ? 'read' : loaded ? 'strip' : 'plain');
		const bsx = sx + T * scell + GAP;
		out.push(svg('text', { x: sx, y: sy - 6, class: 'sim-label' }, ['As']));
		grid(out, sx, sy, T, T, scell, (_r, q) => mark(compute && q === x.k), (r, q) => (loaded ? String(st.asReads[r * T + q]) : ''));
		out.push(svg('text', { x: bsx, y: sy - 6, class: 'sim-label' }, ['Bs']));
		grid(out, bsx, sy, T, T, scell, (r) => mark(compute && r === x.k), (r, q) => (loaded ? String(st.bsReads[r * T + q]) : ''));
		const caption = loaded ? `phase ${st.tileA!.c0 / T}, reads this phase` : 'shared memory, empty';
		out.push(svg('text', { x: sx, y: sy + T * scell + 16, class: 'sim-small' }, [caption]));
		height = Math.max(height, sy + T * scell + 20);
	}

	height += PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
