import { clear, svg } from '@sim-core/svg';
import { PATHS, type PathId } from './model';

const PAD = 8;
const BOX_MIN = 100;
const BOX_H = 52;
const GAP = 22;
const ROW_GAP = 30;

/** Box positions, wrapping into rows so no box is narrower than BOX_MIN. */
export function layout(count: number, width: number) {
	const cols = Math.max(1, Math.min(count, Math.floor((width - 2 * PAD + GAP) / (BOX_MIN + GAP))));
	const boxW = (width - 2 * PAD - (cols - 1) * GAP) / cols;
	const rows = Math.ceil(count / cols);
	const boxes = Array.from({ length: count }, (_, i) => ({
		x: PAD + (i % cols) * (boxW + GAP),
		y: PAD + Math.floor(i / cols) * (BOX_H + ROW_GAP),
	}));
	return { boxes, boxW, height: 2 * PAD + rows * BOX_H + (rows - 1) * ROW_GAP };
}

export function render(root: SVGSVGElement, path: PathId, step: number, width: number, onSelect: (i: number) => void) {
	const stages = PATHS[path].stages;
	const { boxes, boxW, height } = layout(stages.length, width);
	const nodes: SVGElement[] = [
		svg('defs', {}, [
			svg('marker', { id: 'pipe-arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto' }, [
				svg('path', { d: 'M0,0 L8,4 L0,8 z', class: 'pipe-arrowhead' }),
			]),
		]),
	];

	for (let i = 0; i + 1 < stages.length; i++) {
		const a = boxes[i];
		const b = boxes[i + 1];
		const d =
			a.y === b.y
				? `M${a.x + boxW},${a.y + BOX_H / 2} H${b.x - 2}`
				: `M${a.x + boxW / 2},${a.y + BOX_H} V${a.y + BOX_H + ROW_GAP / 2} H${b.x + boxW / 2} V${b.y - 2}`;
		nodes.push(svg('path', { d, class: `pipe-link${i < step ? ' is-done' : ''}`, 'marker-end': 'url(#pipe-arrow)' }));
	}

	stages.forEach((s, i) => {
		const { x, y } = boxes[i];
		const state = i === step ? ' is-current' : i < step ? ' is-done' : '';
		const g = svg('g', { class: `pipe-stage${state}${s.commands ? ' is-command' : ''}` }, [
			svg('rect', { x, y, width: boxW, height: BOX_H, rx: 4, class: 'pipe-box' }),
			svg('text', { x: x + 10, y: y + 21, class: 'sim-label' }, [s.label]),
			svg('text', { x: x + 10, y: y + 39, class: 'sim-small' }, [s.tool]),
		]);
		g.addEventListener('click', () => onSelect(i));
		nodes.push(g);
	});

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes);
}
