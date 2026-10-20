import { clear, svg } from '@sim-core/svg';
import { isAdd, steps, WARP, warpLanes, warpState, type Config, type Step } from './model';

const PAD = 8;
const GAP = 2;
const LABEL_H = 16;
const ARC_H = 18;
const STRIP_H = 4;
const ROW_GAP = 8;
const CHAR_W = 5.6;
// Approximate advance of the 13px and 11px mono fonts used for labels.
const LABEL_CHAR = 7.9;
const SMALL_CHAR = 6.7;

/** 32 cells per row when they are at least 12px wide, else 16. */
const cellsPerRow = (inner: number) => ((inner - 31 * GAP) / 32 >= 12 ? 32 : 16);

interface Cell {
	x: number;
	y: number;
	size: number;
}

const text = (x: number, y: number, s: string, cls = 'sim-small', anchor?: string) => svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/** The long label if it fits in `inner`, else the short one. */
const fit = (inner: number, long: string, short: string, cls = 'sim-small') =>
	long.length * (cls === 'sim-label' ? LABEL_CHAR : SMALL_CHAR) <= inner ? long : short;

function warpStatus(s: Step, w: number, short: boolean): string {
	const lanes = warpLanes(s)[w];
	if (s.kind === 'load') return '32 lanes load';
	if (s.kind === 'write') return w === 0 ? (short ? 'lane 0 stores' : 'lane 0 stores the sum') : 'done';
	if (s.kind === 'shuffle') return w === 0 ? `32/32${short ? '' : ' lanes'}, ${s.stride} used` : 'done';
	const st = warpState(lanes);
	return st === 'idle' ? 'no work' : st === 'full' ? '32/32 lanes' : `${lanes}/32${short ? '' : ' lanes'}, divergent`;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const all = steps(c);
	const s = all[step];
	const inner = width - 2 * PAD;
	const perRow = cellsPerRow(inner);
	const size = Math.min(30, Math.floor((inner - (perRow - 1) * GAP) / perRow));
	const rowH = LABEL_H + ARC_H + size + GAP + STRIP_H + ROW_GAP;
	const nodes: SVGElement[] = [
		svg('defs', {}, [
			svg('marker', { id: 'rd-arrow', viewBox: '0 0 6 6', refX: 5, refY: 3, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, [
				svg('path', { d: 'M0,0 L6,3 L0,6 z', class: 'rd-arrowhead' }),
			]),
		]),
	];

	const levels = all.filter(isAdd).length;
	const level = all.slice(1, step + 1).filter(isAdd).length;
	const head =
		s.kind === 'load'
			? `loaded: ${c.block} values, ${levels} levels to go`
			: s.kind === 'write'
				? `done: out[blockIdx.x] = ${s.out}`
				: s.kind === 'shared'
					? `level ${level} of ${levels}: stride ${s.stride}`
					: s.kind === 'regs'
						? `level ${level} of ${levels}: into registers`
						: `level ${level} of ${levels}: shuffle offset ${s.stride}`;
	nodes.push(text(PAD, 14, fit(inner, head, head.replace(' values, ', ', ').replace('out[blockIdx.x]', 'out'), 'sim-label'), 'sim-label'));
	let y = 28;

	const cells = new Map<string, Cell>();
	const showValues = (v: number) => String(v).length * CHAR_W + 3 <= size;

	function drawSection(key: 'regs' | 'shared', count: number, values: number[] | null, live: boolean[] | null, rowLabel: (row: number) => string) {
		const rows = Math.ceil(count / perRow);
		const dest = new Set<number>();
		const src = new Set<number>();
		for (const [d, from] of s.pairs) {
			const destKey = s.kind === 'shared' ? 'shared' : 'regs';
			const srcKey = s.kind === 'regs' ? 'shared' : destKey;
			if (destKey === key) dest.add(d);
			if (srcKey === key) src.add(from);
		}
		for (let r = 0; r < rows; r++) {
			const top = y + r * rowH;
			nodes.push(text(PAD, top + 11, rowLabel(r)));
			for (let k = 0; k < perRow && r * perRow + k < count; k++) {
				const i = r * perRow + k;
				const cell = { x: PAD + k * (size + GAP), y: top + LABEL_H + ARC_H, size };
				cells.set(`${key}:${i}`, cell);
				const state = !values ? 'is-empty' : dest.has(i) ? 'is-dest' : src.has(i) ? 'is-src' : live?.[i] ? 'is-live' : 'is-dead';
				nodes.push(svg('rect', { x: cell.x, y: cell.y, width: size, height: size, rx: 2, class: `rd-cell ${state}` }));
				if (values && showValues(values[i])) {
					nodes.push(text(cell.x + size / 2, cell.y + size / 2 + 3.5, String(values[i]), `rd-value ${state}`, 'middle'));
				}
				if (key === 'shared') {
					const lane = !s.active[i] ? 'is-off' : s.useful[i] ? 'is-on' : 'is-wasted';
					nodes.push(svg('rect', { x: cell.x, y: cell.y + size + GAP, width: size, height: STRIP_H, class: `rd-lane ${lane}` }));
				}
			}
		}
		y += rows * rowH;
	}

	const narrow = perRow < WARP;
	const half = (r: number) => (narrow ? ` (${(r % 2) * 16}-${(r % 2) * 16 + 15})` : '');
	if (c.mode === 'warp-final') {
		const regsStatus = !s.regs ? 'not used yet' : s.kind === 'write' ? 'lane 0 holds the sum' : s.kind === 'shuffle' ? `${s.stride} of 32 results used` : '32 lanes add 2 values each';
		nodes.push(text(PAD, y + 4, 'registers v of warp 0', 'sim-label'));
		y += 8;
		drawSection('regs', WARP, s.regs, s.regsLive, (r) => (r === 0 ? regsStatus : `lanes ${r * perRow}-${r * perRow + perRow - 1}`));
	}
	nodes.push(text(PAD, y + 4, fit(inner, 'sdata[] in shared memory, one slot per thread', 'sdata[] in shared memory', 'sim-label'), 'sim-label'));
	y += 8;
	drawSection('shared', c.block, s.shared, s.sharedLive, (r) => {
		const w = Math.floor((r * perRow) / WARP);
		const first = perRow === WARP || r % 2 === 0;
		return first ? `warp ${w}${half(r)}: ${warpStatus(s, w, narrow)}` : `warp ${w}${half(r)}`;
	});

	// Links: from the value read to the slot it is added into, drawn under the cells and labels.
	const links: SVGElement[] = [];
	const srcKey = s.kind === 'regs' ? 'shared' : s.kind === 'shared' ? 'shared' : 'regs';
	const destKey = s.kind === 'shared' ? 'shared' : 'regs';
	for (const [d, from] of s.pairs) {
		const a = cells.get(`${srcKey}:${from}`);
		const b = cells.get(`${destKey}:${d}`);
		if (!a || !b || (a === b)) continue;
		const ax = a.x + size / 2;
		const bx = b.x + size / 2;
		if (a.y === b.y) {
			const h = Math.min(ARC_H - 3, 4 + Math.abs(bx - ax) / 6);
			links.push(svg('path', { d: `M${ax},${a.y} Q${(ax + bx) / 2},${a.y - 2 * h} ${bx},${b.y}`, class: 'rd-link', 'marker-end': 'url(#rd-arrow)' }));
		} else {
			const up = a.y > b.y;
			links.push(svg('line', { x1: ax, y1: up ? a.y : a.y + size, x2: bx, y2: up ? b.y + size + (destKey === 'shared' ? GAP + STRIP_H : 0) : b.y, class: 'rd-link', 'marker-end': 'url(#rd-arrow)' }));
		}
	}

	nodes.splice(1, 0, ...links);
	const height = y + PAD - ROW_GAP;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes);
}
