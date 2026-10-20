import { clear, svg } from '@sim-core/svg';
import { fmt, inputs, N, PASSES, steps, type Config } from './model';

const PAD = 8;
const GAP = 3;
const DOT_H = 10;
const ROW_GAP = 10;
const BAR_H = 34;
// Approximate advance of the 10px mono font used for cell values, and of the 9px one in narrow cells.
const VALUE_CHAR = 6.1;
const TIGHT_CHAR = 5.5;
const TIGHT = 32;

const PASS_LABEL = { max: 'max', sum: 'sum', stats: 'max and sum', normalize: 'normalize' } as const;

const text = (x: number, y: number, s: string, cls = 'sim-small', anchor?: string) => svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/** Reads of element i after this step: one per finished pass, plus the current one if it covered i. */
function readCount(c: Config, step: number, i: number): number {
	const s = steps(c)[step];
	if (s.kind === 'start') return 0;
	if (s.kind === 'done') return PASSES[c.mode].length;
	return s.pass + (i < s.hi ? 1 : 0);
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const s = steps(c)[step];
	const x = inputs(c.preset);
	const inner = width - 2 * PAD;
	const perRow = (inner - 15 * GAP) / 16 >= 30 ? 16 : 8;
	const size = Math.min(44, Math.floor((inner - (perRow - 1) * GAP) / perRow));
	const rows = N / perRow;
	const nodes: SVGElement[] = [];
	const passes = PASSES[c.mode].length;

	const head =
		s.kind === 'start'
			? `${passes} passes to go`
			: s.kind === 'done'
				? `done after ${passes} passes`
				: `pass ${s.pass + 1} of ${passes}: ${PASS_LABEL[s.kind]}`;
	nodes.push(text(PAD, 14, head, 'sim-label'));
	let y = 26;

	const tight = size < TIGHT ? ' is-tight' : '';
	const fits = (str: string) => str.length * (tight ? TIGHT_CHAR : VALUE_CHAR) + 3 <= size;
	const cellX = (i: number) => PAD + (i % perRow) * (size + GAP);
	const overflow = new Set(s.overflow);
	const reading = (i: number) => s.kind !== 'start' && s.kind !== 'done' && i >= s.lo && i < s.hi;

	// Input row: chunks alternate two tints, the chunk being read gets the accent edge.
	nodes.push(text(PAD, y + 10, perRow === 16 ? 'x: input row, read once per pass (dots)' : 'x: input row', 'sim-small'));
	y += 16;
	for (let r = 0; r < rows; r++) {
		const top = y + r * (size + DOT_H + ROW_GAP);
		for (let k = 0; k < perRow; k++) {
			const i = r * perRow + k;
			const tint = Math.floor(i / c.chunk) % 2 ? 'is-b' : 'is-a';
			const state = overflow.has(i) ? 'is-overflow' : reading(i) ? 'is-reading' : '';
			nodes.push(svg('rect', { x: cellX(i), y: top, width: size, height: size, rx: 2, class: `os-cell ${tint} ${state}` }));
			const v = fmt(x[i]);
			if (fits(v)) nodes.push(text(cellX(i) + size / 2, top + size / 2 + 3.5, v, `os-value ${state}${tight}`, 'middle'));
			const reads = readCount(c, step, i);
			for (let d = 0; d < reads; d++) {
				nodes.push(svg('circle', { cx: cellX(i) + 4 + d * 6, cy: top + size + 5, r: 2, class: 'os-dot' }));
			}
		}
	}
	y += rows * (size + DOT_H + ROW_GAP);

	// Output row: a bar per probability, NaN marked in the warning color.
	nodes.push(text(PAD, y + 10, perRow === 16 ? 'y: probabilities written so far' : 'y: probabilities', 'sim-small'));
	y += 16;
	for (let r = 0; r < rows; r++) {
		const top = y + r * (BAR_H + 14 + ROW_GAP);
		for (let k = 0; k < perRow; k++) {
			const i = r * perRow + k;
			const v = s.y[i];
			const nan = v !== null && Number.isNaN(v);
			nodes.push(svg('rect', { x: cellX(i), y: top, width: size, height: BAR_H, rx: 2, class: `os-slot ${v === null ? 'is-empty' : ''} ${nan ? 'is-nan' : ''}` }));
			if (v !== null && !nan && v > 0) {
				const h = Math.max(1, Math.round(v * BAR_H));
				nodes.push(svg('rect', { x: cellX(i), y: top + BAR_H - h, width: size, height: h, class: 'os-bar' }));
			}
			if (v !== null) {
				const label = nan ? 'NaN' : v.toFixed(2);
				if (fits(label)) nodes.push(text(cellX(i) + size / 2, top + BAR_H + 11, label, `os-yvalue ${nan ? 'is-nan' : ''}${tight}`, 'middle'));
			}
		}
	}
	y += rows * (BAR_H + 14 + ROW_GAP);

	const height = y + PAD - ROW_GAP;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes);
}

