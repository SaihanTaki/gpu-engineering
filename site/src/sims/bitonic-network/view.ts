import { clear, svg } from '@sim-core/svg';
import { stages, type Config, type State } from './model';

const PAD = 12;
const KEY_W = 26;
const BAR_W = 34;
const TOP = 34;

export function render(root: SVGSVGElement, c: Config, st: State, width: number) {
	const n = c.n;
	const all = stages(n);
	const rowH = n === 8 ? 32 : 24;
	const netX = PAD + KEY_W + BAR_W + 14;
	const netW = width - netX - PAD;
	const colW = netW / all.length;
	const wireY = (w: number) => TOP + w * rowH + rowH / 2;
	const bottom = TOP + n * rowH;
	const moved = new Set<number>();
	st.stage?.comparators.forEach((cmp, t) => {
		if (st.swapped[t]) moved.add(cmp.i).add(cmp.p);
	});

	const out: SVGElement[] = [
		svg('text', { x: PAD, y: 14, class: 'sim-small' }, ['keys']),
		svg('text', { x: netX, y: 14, class: 'sim-small' }, ['stage']),
	];

	for (let w = 0; w < n; w++) {
		const y = wireY(w);
		const key = st.keys[w];
		const cls = moved.has(w) ? 'bn-key is-moved' : 'bn-key';
		out.push(
			svg('rect', { x: PAD, y: y - rowH / 2 + 3, width: KEY_W, height: rowH - 6, rx: 3, class: cls }),
			svg('text', { x: PAD + KEY_W / 2, y: y + 4, class: 'sim-cell-text' }, [String(key)]),
			svg('rect', { x: PAD + KEY_W + 4, y: y - 4, width: (BAR_W * key) / n, height: 8, class: 'bn-bar' }),
			svg('line', { x1: netX - 4, y1: y, x2: width - PAD, y2: y, class: 'bn-wire' }),
		);
	}

	all.forEach((stage, s) => {
		const x0 = netX + s * colW;
		const state = s + 1 < st.step ? 'is-done' : s + 1 === st.step ? 'is-current' : 'is-future';
		out.push(svg('text', { x: x0 + colW / 2, y: 28, class: `sim-index${state === 'is-current' ? ' bn-stage-now' : ''}` }, [String(s + 1)]));
		if (s > 0 && stage.j === stage.k / 2) {
			out.push(svg('line', { x1: x0, y1: TOP, x2: x0, y2: bottom, class: 'bn-phase' }));
		}
		stage.comparators.forEach((cmp, t) => {
			// Comparators of one stage that overlap vertically get their own offset inside the column.
			const lane = t % stage.j;
			const x = x0 + colW * (0.5 + (stage.j > 1 ? (lane / (stage.j - 1) - 0.5) * 0.6 : 0));
			const y1 = wireY(cmp.i);
			const y2 = wireY(cmp.p);
			const swap = state === 'is-current' && st.swapped[t] ? ' is-swap' : '';
			const g = svg('g', { class: `bn-cmp ${state}${swap}` }, [
				svg('line', { x1: x, y1, x2: x, y2 }),
				svg('circle', { cx: x, cy: cmp.up ? y1 : y2, r: 2.5 }),
			]);
			// The arrowhead marks the wire that receives the larger key.
			const hy = cmp.up ? y2 : y1;
			const d = cmp.up ? 1 : -1;
			g.append(svg('path', { d: `M${x - 3.5} ${hy - 6 * d} L${x + 3.5} ${hy - 6 * d} L${x} ${hy} Z` }));
			out.push(g);
		});
	});

	// Phase labels: the stages that share one k.
	out.push(svg('text', { x: PAD, y: bottom + 18, class: 'sim-small' }, ['k =']));
	let s = 0;
	while (s < all.length) {
		const k = all[s].k;
		let e = s;
		while (e < all.length && all[e].k === k) e++;
		out.push(svg('text', { x: netX + ((s + e) / 2) * colW, y: bottom + 18, class: 'sim-index' }, [String(k)]));
		s = e;
	}

	const height = bottom + 28;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
