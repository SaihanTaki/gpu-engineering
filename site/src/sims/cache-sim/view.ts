import { clear, svg } from '@sim-core/svg';
import type { Access, Config, Outcome } from './model';

const PAD = 12;
const GAP = 3;
const MIN_CELL = 36;
const CELL_H = 26;

const text = (x: number, y: number, s: string, cls = 'sim-label') => svg('text', { x, y, class: cls }, [s]);

/** Bar height in the timeline: the farther the data came from, the taller the bar. */
const BAR: Record<Outcome, number> = { l1: 0.3, l2: 0.6, dram: 1 };

export function render(root: SVGSVGElement, c: Config, sim: Access[], step: number, width: number) {
	const inner = width - 2 * PAD;
	const cur = step > 0 ? sim[step - 1] : null;
	const nodes: SVGElement[] = [];
	let y = PAD;

	const cache = (name: string, lines: number[], capacity: number, evicted: number | null, mark: 'hit' | 'filled' | null) => {
		nodes.push(text(PAD, y + 12, `${name}: ${lines.length} / ${capacity} lines`));
		y += 20;
		const cols = Math.max(1, Math.min(capacity, Math.floor((inner + GAP) / (MIN_CELL + GAP))));
		const w = Math.min(56, (inner - (cols - 1) * GAP) / cols);
		for (let i = 0; i < capacity; i++) {
			const x = PAD + (i % cols) * (w + GAP);
			const top = y + Math.floor(i / cols) * (CELL_H + GAP);
			const line = lines[i];
			let cls = 'cache-slot';
			if (line === undefined) cls += ' is-empty';
			// The line just used is always the most recent, slot 0.
			else if (i === 0 && mark) cls += ` is-${mark}`;
			nodes.push(svg('rect', { x, y: top, width: w, height: CELL_H, rx: 2, class: cls }));
			const on = i === 0 && mark === 'hit' ? ' is-on' : '';
			if (line !== undefined) nodes.push(svg('text', { x: x + w / 2, y: top + 17, class: `sim-cell-text cache-slot-text${on}` }, [String(line)]));
		}
		y += Math.ceil(capacity / cols) * (CELL_H + GAP) - GAP;
		// The row is kept when empty so the drawing does not jump between steps.
		if (evicted !== null) nodes.push(text(PAD, y + 16, `Evicted line ${evicted} (LRU)`, 'sim-small cache-evicted'));
		y += 34;
	};

	const o = cur?.outcome;
	cache('L1', cur?.l1 ?? [], c.l1Lines, cur?.l1Evicted ?? null, !o ? null : o === 'l1' ? 'hit' : 'filled');
	// An L1 hit never reaches L2, so nothing is marked there.
	cache('L2', cur?.l2 ?? [], c.l2Lines, cur?.l2Evicted ?? null, o === 'l2' ? 'hit' : o === 'dram' ? 'filled' : null);

	nodes.push(text(PAD, y + 12, `Accesses 1 to ${sim.length}`));
	y += 20;
	const barH = 34;
	const bw = inner / sim.length;
	nodes.push(svg('rect', { x: PAD, y, width: inner, height: barH, class: 'cache-track' }));
	for (let i = 0; i < step; i++) {
		const h = barH * BAR[sim[i].outcome];
		nodes.push(svg('rect', { x: PAD + i * bw, y: y + barH - h, width: Math.max(0.5, bw - (bw > 3 ? 1 : 0)), height: h, class: `cache-bar is-${sim[i].outcome}` }));
	}
	if (cur) nodes.push(svg('rect', { x: PAD + (step - 1) * bw - 1, y: y - 3, width: Math.max(2, bw) + 2, height: barH + 6, class: 'cache-now' }));
	y += barH + PAD;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
