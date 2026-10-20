import { clear, svg } from '@sim-core/svg';
import { simulate, type Config, type WarpState } from './model';

const PAD = 12;
const LABEL_W = 50;
const CELL = 16;
const ROW = 22;
const TOP = 30;

const LETTER: Partial<Record<WarpState, string>> = { load: 'L', alu: 'A' };

/** Draws the last cycles that fit in `width`, ending at the cycle that ran in `step`. */
export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const trace = simulate(c);
	const cols = Math.max(4, Math.floor((width - 2 * PAD - LABEL_W) / CELL));
	const end = Math.min(step, trace.length);
	const start = Math.max(0, end - cols);
	const shown = trace.slice(start, end);
	const x0 = PAD + LABEL_W;
	const out: SVGElement[] = [
		svg('defs', {}, [
			svg('pattern', { id: 'ws-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 5, height: 5, class: 'ws-hatch-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'ws-hatch-line' }),
			]),
		]),
		svg('text', { x: PAD, y: 14, class: 'sim-label' }, [end ? `cycles ${start} to ${end - 1}` : 'before cycle 0']),
	];

	const rowY = (r: number) => TOP + r * ROW;
	out.push(svg('text', { x: PAD, y: rowY(0) + 13, class: 'sim-small' }, ['issued']));
	for (let w = 0; w < c.warps; w++) out.push(svg('text', { x: PAD, y: rowY(w + 1) + 13, class: 'sim-small' }, [`warp ${w}`]));

	shown.forEach((cy, k) => {
		const x = x0 + k * CELL;
		const current = start + k === end - 1;
		const idle = cy.issued === null;
		out.push(svg('rect', { x: x + 1, y: rowY(0) + 1, width: CELL - 2, height: ROW - 4, rx: 2, class: `ws-slot${idle ? ' is-idle' : ''}` }));
		if (!idle) out.push(svg('text', { x: x + CELL / 2, y: rowY(0) + 13, class: 'ws-cell-text' }, [String(cy.issued)]));
		cy.warps.forEach((st, w) => {
			if (st === 'done') return;
			const y = rowY(w + 1);
			out.push(svg('rect', { x: x + 1, y: y + 1, width: CELL - 2, height: ROW - 4, rx: 2, class: `ws-cell is-${st}`, fill: st === 'waiting' ? 'url(#ws-hatch)' : undefined }));
			const letter = LETTER[st];
			if (letter) out.push(svg('text', { x: x + CELL / 2, y: y + 13, class: `ws-cell-text is-${st}` }, [letter]));
		});
		if (current) out.push(svg('rect', { x: x, y: rowY(0) - 1, width: CELL, height: (c.warps + 1) * ROW, rx: 3, class: 'ws-now' }));
		if (cy.cycle % 10 === 0) out.push(svg('text', { x: x + CELL / 2, y: rowY(c.warps + 1) + 12, class: 'sim-index' }, [String(cy.cycle)]));
	});

	const height = rowY(c.warps + 1) + 20 + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
