import { clear, svg } from '@sim-core/svg';
import { ALGO_IDS, ALGO_NAMES, countsAt, depth, inputOf, phases, work, type AlgoId, type Config } from './model';

const PAD = 12;
const COL_GAP = 24;
const CELL_H = 20;
const ROW_GAP = 14;
const HEAD_H = 50;
const LABEL_W = 44;

const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) => svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

interface Box {
	x: number;
	y: number;
	w: number;
}

/** Draws one algorithm's rows: the input, then one row per phase, filled up to the current step. */
function drawAlgo(nodes: SVGElement[], c: Config, algo: AlgoId, step: number, box: Box): number {
	const ps = phases(c, algo);
	const k = countsAt(c, algo, step);
	const labelW = (box.w - LABEL_W) / c.n >= 18 ? LABEL_W : 0;
	const cellW = Math.min(36, (box.w - labelW) / c.n);
	const font = cellW >= 16 ? 'sc-val' : 'sc-val is-tight';
	const cx = (i: number) => box.x + labelW + i * cellW + cellW / 2;
	const rowY = (r: number) => box.y + HEAD_H + r * (CELL_H + ROW_GAP);

	const current = step === 0 ? 'input' : step > ps.length ? 'done' : `now: ${ps[step - 1].label}`;
	nodes.push(text(box.x, box.y + 13, ALGO_NAMES[algo]));
	nodes.push(text(box.x, box.y + 29, current, step > 0 && step <= ps.length ? 'sim-small sc-now' : 'sim-small'));
	nodes.push(text(box.x, box.y + 43, `${k.adds} adds, ${k.addLevels} of ${depth[algo](c.n)} levels`, 'sim-small'));

	const rows = [{ label: 'in', values: inputOf(c), changed: [] as number[] }, ...ps];
	rows.forEach((row, r) => {
		const y = rowY(r);
		const shown = r <= step;
		const isNow = r === step && r > 0;
		if (labelW) nodes.push(text(box.x, y + 14, row.label, isNow ? 'sim-small sc-now' : 'sim-small'));
		if (r > 0) {
			// Lines from the cells read in the row above to the cells written in this row.
			for (const op of ps[r - 1].ops) {
				if (!shown) continue;
				nodes.push(
					svg('line', {
						x1: cx(op.from),
						y1: y - ROW_GAP,
						x2: cx(op.to),
						y2: y,
						class: isNow ? 'sc-edge is-now' : 'sc-edge',
					}),
				);
			}
		}
		const changed = new Set(row.changed);
		for (let i = 0; i < c.n; i++) {
			const x = box.x + labelW + i * cellW;
			const cls = !shown ? 'sc-cell is-future' : isNow && changed.has(i) ? 'sc-cell is-now' : 'sc-cell';
			nodes.push(svg('rect', { x: x + 1, y, width: cellW - 2, height: CELL_H, rx: 2, class: cls }));
			if (shown) nodes.push(text(x + cellW / 2, y + 14, String(row.values[i]), font, 'middle'));
		}
	});
	return rowY(rows.length) - ROW_GAP;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const half = (inner - COL_GAP) / 2;
	// Side by side only while each half still has labelled cells of at least 24px.
	const sideBySide = (half - LABEL_W) / c.n >= 24;
	const colW = sideBySide ? half : inner;
	const nodes: SVGElement[] = [];
	let y = PAD;
	let bottom = y;
	ALGO_IDS.forEach((algo, i) => {
		const box = sideBySide ? { x: PAD + i * (colW + COL_GAP), y: PAD, w: colW } : { x: PAD, y, w: colW };
		bottom = Math.max(bottom, drawAlgo(nodes, c, algo, step, box));
		if (!sideBySide) y = bottom + 20;
	});
	bottom += 18;
	nodes.push(text(PAD, bottom, `Sequential loop: ${work.sequential(c.n)} adds, ${depth.sequential(c.n)} steps`, 'sim-small'));
	const h = bottom + PAD;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${h}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(h));
	root.append(...nodes);
}
