import { clear, svg } from '@sim-core/svg';
import { endTime, fmt, laneAt, lanes, phaseText, timeAt, type Config, type Lane, type LaneState } from './model';

const PAD = 12;
const BAR_H = 14;

const TITLES = {
	cpu: (c: Config) => `CPU: ${c.cpuWorkers} fast worker${c.cpuWorkers === 1 ? '' : 's'}`,
	gpu: (c: Config) => `GPU: ${fmt(c.gpuWorkers)} slow workers`,
	split: () => 'CPU serial + GPU parallel',
};

/** Few workers get big cells; a thousand get a dense grid about 120px tall. */
function gridShape(n: number, inner: number) {
	const gap = n <= 64 ? 4 : 1;
	const cell =
		n <= 64
			? Math.max(8, Math.min(28, Math.floor((inner + gap) / n) - gap))
			: Math.max(3, Math.min(12, Math.floor(Math.sqrt((inner * 120) / n)) - gap));
	const cols = Math.max(1, Math.floor((inner + gap) / (cell + gap)));
	return { gap, cell, cols, height: Math.ceil(n / cols) * (cell + gap) - gap };
}

function bar(l: Lane, st: LaneState, x: number, y: number, w: number) {
	const total = l.serialUnits + l.parallelUnits;
	const serialW = (w * l.serialUnits) / total;
	const parallelW = w - serialW;
	const nodes: SVGElement[] = [svg('rect', { x, y, width: w, height: BAR_H, rx: 2, class: 'sim-race-track' })];
	if (l.serialUnits > 0) {
		nodes.push(
			svg('rect', { x, y, width: (serialW * st.serialDone) / l.serialUnits, height: BAR_H, class: 'sim-race-serial' }),
		);
	}
	if (l.parallelUnits > 0) {
		nodes.push(
			svg('rect', {
				x: x + serialW,
				y,
				width: (parallelW * st.parallelDone) / l.parallelUnits,
				height: BAR_H,
				class: 'sim-race-parallel',
			}),
		);
	}
	if (l.serialUnits > 0 && l.parallelUnits > 0) {
		nodes.push(svg('line', { x1: x + serialW, y1: y - 3, x2: x + serialW, y2: y + BAR_H + 3, class: 'sim-race-divider' }));
	}
	return nodes;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const t = timeAt(c, step);
	const out: SVGElement[] = [
		// Stripes keep the serial part distinct from the parallel part without relying on color.
		svg('defs', {}, [
			svg('pattern', { id: 'sim-race-stripes', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 5, height: 5, class: 'sim-race-stripe-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'sim-race-stripe' }),
			]),
		]),
		svg('text', { x: PAD, y: 18, class: 'sim-label' }, [`t = ${fmt(t)} of ${fmt(endTime(c))} ticks`]),
	];
	let y = 34;

	for (const l of lanes(c)) {
		const st = laneAt(l, t);
		out.push(
			svg('text', { x: PAD, y: y + 13, class: 'sim-label' }, [TITLES[l.id](c)]),
			svg('text', { x: PAD, y: y + 29, class: 'sim-small' }, [`${Math.floor(st.fraction * 100)}%, ${phaseText(l, st)}`]),
			...bar(l, st, PAD, y + 38, inner),
		);
		y += 38 + BAR_H + 10;
		if (l.id === 'split') continue;

		const g = gridShape(l.workers, inner);
		const workers: SVGElement[] = [];
		// Stripes do not show in cells this small, so tiny serial cells get a solid fill.
		const serialClass = g.cell >= 8 ? 'sim-worker is-busy is-serial' : 'sim-worker is-busy is-serial is-tiny';
		const busyClass = st.phase === 'serial' ? serialClass : 'sim-worker is-busy';
		for (let i = 0; i < l.workers; i++) {
			workers.push(
				svg('rect', {
					x: PAD + (i % g.cols) * (g.cell + g.gap),
					y: y + Math.floor(i / g.cols) * (g.cell + g.gap),
					width: g.cell,
					height: g.cell,
					rx: g.cell >= 8 ? 2 : 0,
					class: i < st.busy ? busyClass : 'sim-worker',
				}),
			);
		}
		out.push(svg('g', {}, workers));
		y += g.height + 20;
	}

	const height = y - 10 + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
