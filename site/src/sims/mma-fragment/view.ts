import { clear, svg } from '@sim-core/svg';
import { groupId, ownerOf, SHAPE, viewAt, type Config, type MatrixId } from './model';

const GAP = 18;
const PAD = 2;
const TITLE_H = 18;
const INDEX_H = 13;

export type PickHandler = (m: MatrixId, row: number, col: number) => void;

const text = (x: number, y: number, s: string, cls: string, anchor?: string) => svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/**
 * A (16 x 16), B (16 x 8) and C/D (16 x 8), each cell printed with the lane
 * that holds it. Wide: the three side by side. Narrow: A on top, B and C below.
 */
export function render(root: SVGSVGElement, c: Config, step: number, width: number, onPick: PickHandler) {
	const wide = width >= 600;
	const cols = wide ? 32 : 16;
	// Narrow: B and C share a row below A, with one gap between them.
	const gaps = wide ? 2 * GAP : GAP;
	const cell = Math.max(12, Math.min(26, Math.floor((width - 2 * PAD - gaps - 16) / cols)));
	const v = viewAt(c, step);
	const nodes: SVGElement[] = [];
	const marks: SVGElement[] = [];
	const showIndex = cell >= 15;
	const top = TITLE_H + (showIndex ? INDEX_H : 0);
	const left = PAD + 16;

	const origin: Record<MatrixId, { x: number; y: number }> = wide
		? {
				a: { x: left, y: top },
				b: { x: left + 16 * cell + GAP, y: top },
				c: { x: left + 24 * cell + 2 * GAP, y: top },
			}
		: {
				a: { x: left, y: top },
				b: { x: left, y: top + 16 * cell + top + 8 },
				c: { x: left + 8 * cell + GAP, y: top + 16 * cell + top + 8 },
			};
	const height = (wide ? top + 16 * cell : 2 * (top + 16 * cell) + 8) + 6;

	const titles: Record<MatrixId, string> = {
		a: cell >= 20 ? 'A, 16 x 16 (row, k)' : 'A 16x16',
		b: cell >= 20 ? 'B, 16 x 8 (k, col)' : 'B 16x8',
		c: cell >= 20 ? 'C and D, 16 x 8' : 'C/D 16x8',
	};

	for (const m of ['a', 'b', 'c'] as MatrixId[]) {
		const o = origin[m];
		const { rows, cols: nc } = SHAPE[m];
		const shown = v.shown.includes(m);
		nodes.push(text(o.x, o.y - (showIndex ? INDEX_H : 0) - 5, titles[m], 'sim-label mf-title'));
		if (showIndex) {
			for (let j = 0; j < nc; j++) nodes.push(text(o.x + j * cell + cell / 2, o.y - 3, String(j), 'sim-index'));
		}
		if (m !== 'c' || !wide) {
			for (let i = 0; i < rows; i++) nodes.push(text(o.x - 3, o.y + i * cell + cell / 2 + 3.5, String(i), 'sim-index mf-row-index', 'end'));
		}
		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < nc; j++) {
				const own = ownerOf(m, i, j);
				const mine = own.lane === c.lane;
				const picked = c.pick?.m === m && c.pick.row === i && c.pick.col === j;
				const isTarget = m === 'c' && v.target?.row === i && v.target.col === j;
				const cls = [
					'mf-cell',
					shown ? (groupId(own.lane) % 2 ? 'is-b' : 'is-a') : 'is-dim',
					mine && shown ? 'is-lane' : '',
					picked ? 'is-picked' : '',
					isTarget ? 'is-target' : '',
				]
					.filter(Boolean)
					.join(' ');
				const x = o.x + j * cell;
				const y = o.y + i * cell;
				const g = svg('g', { class: 'mf-hit' }, [
					svg('rect', { x, y, width: cell, height: cell, class: cls }),
					text(x + cell / 2, y + cell / 2 + (cell < 15 ? 3 : 3.5), String(own.lane), `mf-lane${mine && shown ? ' is-lane' : ''}${shown ? '' : ' is-dim'}${cell < 15 ? ' is-small' : ''}`, 'middle'),
				]);
				g.addEventListener('click', () => onPick(m, i, j));
				nodes.push(g);
			}
		}
	}

	// Steps 4 to 7: the row of A and the column of B that feed the target output.
	if (v.target) {
		const a = origin.a;
		const b = origin.b;
		marks.push(svg('rect', { x: a.x, y: a.y + v.target.row * cell, width: 16 * cell, height: cell, class: 'mf-band' }));
		marks.push(svg('rect', { x: b.x + v.target.col * cell, y: b.y, width: cell, height: 16 * cell, class: 'mf-band' }));
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes, ...marks);
}
