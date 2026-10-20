import { clear, svg } from '@sim-core/svg';
import { WARP, lastStep, tidOf, type Config, type State } from './model';

const PAD = 12;
const GAP = 4;
const BAR_H = 14;

/** 32, 16 or 8 lanes per row, whichever is widest with cells of at least 26px. */
const lanesPerRow = (inner: number) => [32, 16, 8].find((n) => (inner - (n - 1) * GAP) / n >= 26) ?? 8;

export function render(root: SVGSVGElement, c: Config, st: State, width: number) {
	const inner = width - 2 * PAD;
	const perRow = lanesPerRow(inner);
	const cell = Math.min(40, Math.floor((inner - (perRow - 1) * GAP) / perRow));
	const rowH = cell + 30;
	const top = 26;
	const instr = st.instr;
	const out: SVGElement[] = [
		svg('defs', {}, [
			svg('pattern', { id: 'wd-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 5, height: 5, class: 'ws-hatch-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'ws-hatch-line' }),
			]),
		]),
		svg('text', { x: PAD, y: 16, class: 'sim-label' }, [
			instr ? `line ${instr.line}: ${instr.active.filter(Boolean).length} of ${WARP} lanes active` : `warp ${c.warp}: threads ${tidOf(c, 0)} to ${tidOf(c, WARP - 1)}`,
		]),
	];

	for (let lane = 0; lane < WARP; lane++) {
		const x = PAD + (lane % perRow) * (cell + GAP);
		const y = top + Math.floor(lane / perRow) * rowH;
		const masked = instr !== null && !instr.active[lane];
		const cls = instr === null ? 'wd-lane' : masked ? 'wd-lane is-masked' : 'wd-lane is-active';
		out.push(
			svg('text', { x: x + cell / 2, y: y + 10, class: 'sim-index' }, [String(lane)]),
			svg('rect', { x, y: y + 14, width: cell, height: cell, rx: 3, class: cls, fill: masked ? 'url(#wd-hatch)' : undefined }),
		);
		if (st.outcome) {
			out.push(svg('text', { x: x + cell / 2, y: y + 14 + cell / 2 + 4, class: `wd-outcome${masked ? ' is-masked' : ''}` }, [st.outcome[lane] ? 'T' : 'F']));
		}
	}

	// One bar per issued instruction: the filled part is the lanes that did work.
	const rows = Math.ceil(WARP / perRow);
	let y = top + rows * rowH + 14;
	out.push(svg('text', { x: PAD, y, class: 'sim-label' }, ['issued instructions, lanes used']));
	y += 8;
	// Room for every instruction of the run, so the drawing does not grow while stepping.
	const end = y + lastStep(c) * (BAR_H + 4);
	const barX = PAD + 54;
	const barW = inner - 54 - 48;
	for (const x of st.issued) {
		const used = x.active.filter(Boolean).length;
		out.push(
			svg('text', { x: PAD, y: y + 11, class: 'sim-small' }, [`line ${x.line}`]),
			svg('rect', { x: barX, y, width: barW, height: BAR_H, class: 'wd-bar-track', fill: 'url(#wd-hatch)' }),
			svg('rect', { x: barX, y, width: (barW * used) / WARP, height: BAR_H, class: 'wd-bar' }),
			svg('text', { x: barX + barW + 6, y: y + 11, class: 'sim-small' }, [`${used}/32`]),
		);
		y += BAR_H + 4;
	}
	if (!st.issued.length) out.push(svg('text', { x: PAD, y: y + 11, class: 'sim-small' }, ['none yet']));

	const height = end + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
