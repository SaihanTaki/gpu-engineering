import { clear, svg } from '@sim-core/svg';
import type { GpuSpec } from '@sim-core/gpu-specs';
import { fmt, overflowAt, partitionOf, placedAt, usage, type Occupancy, type ResourceId } from './model';

const PAD = 12;
const GAP = 2;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: h, rx: 2, class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label') => svg('text', { x, y, class: cls }, [s]);

/** Fill class for something owned by block b, given how many blocks are placed. */
const owner = (b: number, placed: number) =>
	b >= placed ? 'occ-free' : b === placed - 1 ? 'occ-used is-new' : `occ-used ${b % 2 ? 'is-b' : 'is-a'}`;

function cells(x: number, y: number, inner: number, count: number, maxCell: number) {
	let cols = count;
	while (cols > 8 && (inner - (cols - 1) * GAP) / cols < 9) cols = Math.ceil(cols / 2);
	const size = Math.min(maxCell, Math.floor((inner - (cols - 1) * GAP) / cols));
	const at = (i: number) => ({ x: x + (i % cols) * (size + GAP), y: y + Math.floor(i / cols) * (size + GAP) });
	return { size, at, height: Math.ceil(count / cols) * (size + GAP) - GAP };
}

export function render(root: SVGSVGElement, g: GpuSpec, o: Occupancy, step: number, width: number) {
	const placed = placedAt(o, step);
	const over = overflowAt(o, step);
	const u = usage(g, o, placed);
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	let y = PAD;

	const row = (id: ResourceId, label: string, draw: (top: number) => number) => {
		nodes.push(text(PAD, y + 12, label));
		const top = y + 20;
		const h = draw(top);
		const full = over && o.limiting.includes(id);
		if (full) {
			nodes.push(rect(PAD - 4, top - 4, inner + 8, h + 8, 'occ-full'));
			nodes.push(text(PAD, top + h + 18, 'Full: the next block does not fit here.', 'sim-small occ-full-text'));
		}
		y = top + h + (full ? 30 : 16);
	};

	row('warps', `Warp slots: ${u.warps} / ${g.maxWarpsPerSm}`, (top) => {
		const c = cells(PAD, top, inner, g.maxWarpsPerSm, 22);
		for (let i = 0; i < g.maxWarpsPerSm; i++) {
			const p = c.at(i);
			nodes.push(rect(p.x, p.y, c.size, c.size, owner(Math.floor(i / o.warpsPerBlock), placed)));
		}
		return c.height;
	});

	row('registers', `Registers: ${fmt(u.regs)} / ${fmt(o.regsPerSm)}`, (top) => {
		const labelW = 26;
		const barW = inner - labelW;
		const barH = 16;
		const scale = barW / o.regsPerPartition;
		const filled = new Array(g.partitionsPerSm).fill(0);
		for (let p = 0; p < g.partitionsPerSm; p++) {
			const by = top + p * (barH + 4);
			nodes.push(text(PAD, by + 12, `P${p}`, 'sim-small'), rect(PAD + labelW, by, barW, barH, 'occ-free'));
		}
		for (let i = 0; i < u.warps; i++) {
			const p = partitionOf(g, i);
			const by = top + p * (barH + 4);
			nodes.push(rect(PAD + labelW + filled[p] * o.regsPerWarp * scale, by, o.regsPerWarp * scale - 1, barH, owner(Math.floor(i / o.warpsPerBlock), placed)));
			filled[p]++;
		}
		return g.partitionsPerSm * (barH + 4) - 4;
	});

	const smemLabel = o.smemPerBlock
		? `Shared memory: ${+(u.smem / 1024).toFixed(2)} / ${g.maxSharedKbPerSm} KB`
		: `Shared memory: not used`;
	row('shared', smemLabel, (top) => {
		const barH = 22;
		nodes.push(rect(PAD, top, inner, barH, 'occ-free'));
		const scale = inner / o.smemPerSm;
		for (let b = 0; b < placed; b++) {
			const x = PAD + b * o.smemPerBlock * scale;
			nodes.push(rect(x, top, o.smemPerBlock * scale - 1, barH, owner(b, placed)));
			if (g.reservedSmemPerBlock) nodes.push(rect(x, top, Math.max(1, g.reservedSmemPerBlock * scale - 1), barH, 'occ-reserved'));
		}
		return barH;
	});

	row('blocks', `Block slots: ${placed} / ${g.maxBlocksPerSm}`, (top) => {
		const c = cells(PAD, top, inner, g.maxBlocksPerSm, 28);
		for (let b = 0; b < g.maxBlocksPerSm; b++) {
			const p = c.at(b);
			nodes.push(rect(p.x, p.y, c.size, c.size, owner(b, placed)));
			if (b < placed && c.size >= 14) nodes.push(svg('text', { x: p.x + c.size / 2, y: p.y + c.size / 2 + 4, class: 'sim-cell-text' }, [String(b)]));
		}
		return c.height;
	});

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
