import { clear, svg } from '@sim-core/svg';
import { gridDim, threadOf, uncovered, type Config, type State } from './model';

export interface Selection {
	block: number;
	thread: number;
}

interface Layout {
	width: number;
	height: number;
	cell: number;
	threadPos: (b: number, t: number) => { x: number; y: number };
	elementPos: (i: number) => { x: number; y: number };
	arrayTop: number;
}

const PAD = 12;
const BLOCK_PAD = 4;
const BLOCK_GAP = 10;
const LABEL_H = 18;

function layout(c: Config, width: number, extra: number): Layout {
	const inner = width - 2 * PAD;
	const cell = Math.max(8, Math.min(34, Math.floor((inner - 2 * BLOCK_PAD) / c.blockDim)));
	const blockW = c.blockDim * cell + 2 * BLOCK_PAD;
	const blockH = LABEL_H + cell + BLOCK_PAD;
	const perRow = Math.max(1, Math.floor((inner + BLOCK_GAP) / (blockW + BLOCK_GAP)));
	const blocks = gridDim(c);
	const gridRows = Math.max(1, Math.ceil(blocks / perRow));
	const gridTop = LABEL_H + 6;
	const arrayTop = gridTop + gridRows * (blockH + BLOCK_GAP) + 28;
	const cols = Math.max(1, Math.floor(inner / cell));
	const count = c.n + extra;
	const arrayRows = Math.max(1, Math.ceil(count / cols));
	const arrayRowH = cell + 14;

	return {
		width,
		height: arrayTop + LABEL_H + arrayRows * arrayRowH + PAD,
		cell,
		arrayTop,
		threadPos: (b, t) => ({
			x: PAD + (b % perRow) * (blockW + BLOCK_GAP) + BLOCK_PAD + t * cell,
			y: gridTop + Math.floor(b / perRow) * (blockH + BLOCK_GAP) + LABEL_H,
		}),
		elementPos: (i) => ({
			x: PAD + (i % cols) * cell,
			y: arrayTop + LABEL_H + Math.floor(i / cols) * arrayRowH,
		}),
	};
}

const blockFill = (b: number) => (b % 2 === 0 ? 'var(--sim-block-a)' : 'var(--sim-block-b)');

export function render(
	root: SVGSVGElement,
	c: Config,
	st: State,
	selected: Selection | null,
	width: number,
	onSelect: (s: Selection) => void,
) {
	const blocks = gridDim(c);
	const oobCount = c.guard ? 0 : blocks * c.blockDim - c.n;
	const L = layout(c, width, Math.max(0, oobCount));
	const missing = new Set(uncovered(c));
	const half = L.cell / 2;
	const showText = L.cell >= 20;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${L.width} ${L.height}`);
	root.setAttribute('width', String(L.width));
	root.setAttribute('height', String(L.height));

	root.append(
		svg('defs', {}, [
			svg('pattern', { id: 'sim-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: 'var(--sim-muted)', 'stroke-width': 1.5, 'stroke-opacity': 0.5 }),
			]),
		]),
		svg('text', { x: PAD, y: LABEL_H - 4, class: 'sim-label' }, [
			`grid: gridDim.x = ${blocks}, blockDim.x = ${c.blockDim}`,
		]),
	);

	const links = svg('g', { class: 'sim-links' });
	root.append(links);

	for (let b = 0; b < blocks; b++) {
		const origin = L.threadPos(b, 0);
		const running = st.block === b;
		root.append(
			svg('rect', {
				x: origin.x - BLOCK_PAD,
				y: origin.y - LABEL_H,
				width: c.blockDim * L.cell + 2 * BLOCK_PAD,
				height: LABEL_H + L.cell + BLOCK_PAD,
				rx: 3,
				class: running ? 'sim-block is-running' : 'sim-block',
			}),
			svg('text', { x: origin.x, y: origin.y - 5, class: 'sim-small' }, [`blockIdx.x = ${b}`]),
		);

		for (let t = 0; t < c.blockDim; t++) {
			const th = threadOf(c, b, t);
			const p = L.threadPos(b, t);
			const isSel = selected?.block === b && selected.thread === t;
			const idle = !th.writes || (running && st.phase === 'guard' && !th.inBounds);
			const cell = svg(
				'g',
				{ class: `sim-thread${isSel ? ' is-selected' : ''}`, 'data-b': b, 'data-t': t },
				[
					svg('rect', {
						x: p.x + 1,
						y: p.y + 1,
						width: L.cell - 2,
						height: L.cell - 2,
						rx: 2,
						fill: idle ? 'url(#sim-hatch)' : blockFill(b),
						class: 'sim-cell',
					}),
					showText
						? svg('text', { x: p.x + half, y: p.y + half + 4, class: 'sim-cell-text' }, [String(t)])
						: svg('title', {}, [`threadIdx.x = ${t}`]),
				],
			);
			cell.addEventListener('pointerenter', () => onSelect({ block: b, thread: t }));
			cell.addEventListener('click', () => onSelect({ block: b, thread: t }));
			root.append(cell);

			const linked = isSel || (running && (st.phase !== 'guard' || th.inBounds) && th.writes);
			if (linked && (th.inBounds || !c.guard)) {
				const e = L.elementPos(th.i);
				const x1 = p.x + half;
				const y1 = p.y + L.cell;
				const x2 = e.x + half;
				const y2 = e.y;
				const my = (y1 + y2) / 2;
				links.append(
					svg('path', {
						d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`,
						class: `sim-link${isSel ? ' is-selected' : ''}${th.inBounds ? '' : ' is-oob'}`,
					}),
				);
			}
		}
	}

	root.append(
		svg('text', { x: PAD, y: L.arrayTop + LABEL_H - 4, class: 'sim-label' }, [`out[ ]: n = ${c.n}`]),
	);

	const oob = new Set(st.outOfBounds);
	for (let i = 0; i < c.n + Math.max(0, oobCount); i++) {
		const p = L.elementPos(i);
		const inside = i < c.n;
		const writer = inside ? st.writtenBy[i] : null;
		const cls = !inside
			? `sim-element is-past-end${oob.has(i) ? ' is-hit' : ''}`
			: missing.has(i)
				? 'sim-element is-missing'
				: writer !== null
					? 'sim-element is-written'
					: 'sim-element';
		root.append(
			svg('g', {}, [
				svg('rect', {
					x: p.x + 1,
					y: p.y + 1,
					width: L.cell - 2,
					height: L.cell - 2,
					rx: 2,
					class: cls,
					fill: writer !== null ? blockFill(writer) : undefined,
				}),
				showText || i % 4 === 0
					? svg('text', { x: p.x + half, y: p.y + L.cell + 11, class: 'sim-index' }, [String(i)])
					: svg('title', {}, [`out[${i}]`]),
				writer !== null && showText
					? svg('text', { x: p.x + half, y: p.y + half + 4, class: 'sim-cell-text' }, [String(i)])
					: svg('g'),
			]),
		);
	}
}
