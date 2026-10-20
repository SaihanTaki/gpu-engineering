import { clear, svg } from '@sim-core/svg';
import { fix, inImage, isHalo, readers, roundOf, side, totals, width, type Config, type State } from './model';

const PAD = 12;
const LABEL = 20;
const BAR = 14;

export function render(root: SVGSVGElement, c: Config, st: State, w: number) {
	const inner = w - 2 * PAD;
	const s = side(c);
	const R = c.radius;
	const T = c.tile;
	const cell = Math.max(5, Math.min(28, Math.floor(inner / s)));
	const x0 = PAD;
	const y0 = PAD + LABEL;
	const out: SVGElement[] = [];
	const x = st.cur;
	const counting = st.computed;

	out.push(svg('text', { x: x0, y: y0 - 6, class: 'sim-label' }, [`tile[${s}][${s}] in shared memory`]));
	for (let r = 0; r < s; r++)
		for (let q = 0; q < s; q++) {
			const round = roundOf(c, r, q);
			const region = !inImage(c, r, q) ? 'out' : isHalo(c, r, q) ? 'halo' : 'own';
			const status = x.kind === 'load' && round === x.round ? 'current' : round < st.loadedRounds ? 'loaded' : 'pending';
			const px = x0 + q * cell;
			const py = y0 + r * cell;
			out.push(svg('rect', { x: px, y: py, width: cell, height: cell, class: `hl-cell is-${region} is-${status}` }));
			const text = counting ? String(readers(c, r, q)) : status !== 'pending' ? (region === 'out' ? '0' : String(round + 1)) : '';
			// About 7 px per digit at 11 px mono, plus a margin.
			if (text && cell >= 7 * text.length + 5)
				out.push(svg('text', { x: px + cell / 2, y: py + cell / 2 + 4, class: `sim-cell-text hl-count is-${region} is-${status}` }, [text]));
		}
	// The block's own output tile.
	out.push(svg('rect', { x: x0 + R * cell, y: y0 + R * cell, width: T * cell, height: T * cell, class: 'hl-outline' }));
	if (x.kind === 'compute') {
		const k = width(c) * cell;
		out.push(svg('rect', { x: x0 + 1, y: y0 + 1, width: k - 2, height: k - 2, class: 'hl-window' }));
	}
	let y = y0 + s * cell + 18;
	const caption =
		x.kind === 'compute' || x.kind === 'store'
			? 'numbers: threads that read the slot'
			: st.loadedRounds > 0
				? 'numbers: load round (0 = padding)'
				: 'empty';
	out.push(svg('text', { x: x0, y, class: 'sim-small' }, [caption]));

	// Global loads per output, naive against tiled.
	const t = totals(c);
	y += 24;
	out.push(svg('text', { x: x0, y, class: 'sim-label' }, ['Global loads per output']));
	const bars: [string, number][] = [
		['naive', t.naivePerOutput],
		['tiled', t.tiledPerOutput],
	];
	for (const [name, v] of bars) {
		y += 18;
		out.push(svg('text', { x: x0, y, class: 'sim-small' }, [`${name}: ${fix(v)}`]));
		y += 5;
		out.push(svg('rect', { x: x0, y, width: inner, height: BAR, class: 'hl-bar-track' }));
		out.push(svg('rect', { x: x0, y, width: Math.max(2, (inner * v) / t.naivePerOutput), height: BAR, class: `hl-bar is-${name}` }));
		y += BAR;
	}

	const height = y + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${w} ${height}`);
	root.setAttribute('width', String(w));
	root.setAttribute('height', String(height));
	root.append(...out);
}
