import { clear, svg } from '@sim-core/svg';
import type { GpuSpec } from '@sim-core/gpu-specs';
import { formatKb, levelAt, memoryLevels, perPartition, type Level } from './model';

const PAD = 12;

type Select = (id: string) => void;

/** A clickable group tied to a component id; the HTML part buttons are the keyboard path. */
function part(id: string, selected: string, onSelect: Select, children: SVGElement[]) {
	const g = svg('g', { class: `sim-part${id === selected ? ' is-selected' : ''}`, 'data-part': id }, children);
	g.addEventListener('click', () => onSelect(id));
	g.addEventListener('pointerenter', () => onSelect(id));
	return g;
}

const box = (x: number, y: number, w: number, h: number, cls = 'sim-box') =>
	svg('rect', { x, y, width: w, height: h, rx: 3, class: cls });
const label = (x: number, y: number, text: string, cls = 'sim-label') => svg('text', { x, y, class: cls }, [text]);

function cells(x: number, y: number, count: number, size: number, cols: number, gap = 2, cls = 'sim-unit') {
	const out: SVGElement[] = [];
	for (let i = 0; i < count; i++) {
		out.push(box(x + (i % cols) * (size + gap), y + Math.floor(i / cols) * (size + gap), size, size, cls));
	}
	return { nodes: out, height: Math.ceil(count / cols) * (size + gap) - gap };
}

function chip(g: GpuSpec, w: number, sel: string, on: Select) {
	const inner = w - 2 * PAD - 24;
	const cols = w < 520 ? 12 : 22;
	const gap = 3;
	const size = Math.floor((inner - (cols - 1) * gap) / cols);
	const half = Math.ceil(g.sms / 2);
	const top = cells(PAD + 12, 34, half, size, cols, gap, 'sim-unit is-sm');
	const l2Y = 34 + top.height + 10;
	const bottom = cells(PAD + 12, l2Y + 38, g.sms - half, size, cols, gap, 'sim-unit is-sm');
	const dieH = l2Y + 38 + bottom.height + 12 - 22;
	// SM 0 is the one the next level zooms into.
	(top.nodes[0] as SVGRectElement).classList.add('is-target');

	const memY = 22 + dieH + 14;
	return {
		height: memY + 40 + PAD,
		nodes: [
			box(PAD, 22, w - 2 * PAD, dieH, 'sim-die'),
			label(PAD, 15, `${g.name}, schematic`),
			part('sms', sel, on, [...top.nodes, ...bottom.nodes]),
			part('l2', sel, on, [
				box(PAD + 12, l2Y, inner, 28, 'sim-box is-cache'),
				label(PAD + 22, l2Y + 18, `L2 cache, ${g.l2Mb} MB`),
			]),
			part('dram', sel, on, [
				box(PAD, memY, w - 2 * PAD, 40, 'sim-box is-dram'),
				label(PAD + 10, memY + 25, `${g.memoryGb} GB ${g.memory}, off the die`),
			]),
		],
	};
}

function sm(g: GpuSpec, w: number, sel: string, on: Select) {
	const inner = w - 2 * PAD - 20;
	const pw = (inner - 10) / 2;
	const ph = 92;
	const parts: SVGElement[] = [];
	for (let p = 0; p < g.partitionsPerSm; p++) {
		const x = PAD + 10 + (p % 2) * (pw + 10);
		const y = 34 + Math.floor(p / 2) * (ph + 10);
		parts.push(
			box(x, y, pw, ph, p === 0 ? 'sim-box is-target' : 'sim-box'),
			label(x + 8, y + 18, `Partition ${p}`, 'sim-small'),
			label(x + 8, y + 40, 'warp scheduler', 'sim-small'),
			label(x + 8, y + 58, `${perPartition(g, g.fp32PerSm)} FP32 cores`, 'sim-small'),
			label(x + 8, y + 76, `${perPartition(g, g.tensorPerSm)} tensor core${perPartition(g, g.tensorPerSm) === 1 ? '' : 's'}`, 'sim-small'),
		);
	}
	const warpY = 34 + 2 * (ph + 10) + 22;
	const cols = Math.min(g.maxWarpsPerSm, Math.max(8, Math.floor(inner / 14)));
	const warps = cells(PAD + 10, warpY, g.maxWarpsPerSm, 10, cols, 3, 'sim-unit is-warp');
	const l1Y = warpY + warps.height + 14;
	const smH = l1Y + 36 - 22;
	return {
		height: 22 + smH + PAD,
		nodes: [
			box(PAD, 22, w - 2 * PAD, smH, 'sim-die'),
			label(PAD, 15, 'One SM'),
			part('partitions', sel, on, parts),
			part('warps', sel, on, [label(PAD + 10, warpY - 6, `Warp slots: up to ${g.maxWarpsPerSm} resident warps`, 'sim-small'), ...warps.nodes]),
			part('l1', sel, on, [
				box(PAD + 10, l1Y, inner, 26, 'sim-box is-cache'),
				label(PAD + 20, l1Y + 17, `L1 + shared, ${g.l1SharedKbPerSm} KB`),
			]),
		],
	};
}

function partition(g: GpuSpec, w: number, sel: string, on: Select) {
	const x0 = PAD + 10;
	const inner = w - 2 * PAD - 20;
	const labelW = Math.min(150, inner * 0.38);
	const unitX = x0 + labelW;
	const unitW = inner - labelW;
	const size = 12;
	const cols = Math.max(4, Math.floor((unitW + 3) / (size + 3)));
	const nodes: SVGElement[] = [];
	let y = 34;

	nodes.push(
		part('scheduler', sel, on, [box(x0, y, inner, 28, 'sim-box is-target'), label(x0 + 10, y + 18, 'Warp scheduler')]),
	);
	y += 38;
	const regKb = perPartition(g, g.registerFileKbPerSm);
	nodes.push(part('registers', sel, on, [box(x0, y, inner, 28, 'sim-box is-cache'), label(x0 + 10, y + 18, `Register file, ${regKb} KB`)]));
	y += 42;

	const rows: [string, number, string][] = [
		['fp32', perPartition(g, g.fp32PerSm), 'FP32 core'],
		['int32', perPartition(g, g.int32PerSm), 'INT32 unit'],
		['fp64', perPartition(g, g.fp64PerSm), 'FP64 unit'],
		['tensor', perPartition(g, g.tensorPerSm), 'tensor core'],
		['sfu', perPartition(g, g.sfuPerSm), 'SFU'],
	];
	for (const [id, n, name] of rows) {
		const whole = Number.isInteger(n);
		const big = id === 'tensor';
		const grid = whole
			? cells(unitX, y, n, big ? 26 : size, big ? Math.max(1, Math.floor((unitW + 3) / 29)) : cols, 3, `sim-unit is-${id}`)
			: { nodes: [label(unitX, y + 11, `${g.fp64PerSm} per SM, shared`, 'sim-small')], height: 14 };
		nodes.push(part(id, sel, on, [label(x0, y + 11, whole ? `${n} ${name}${n === 1 ? '' : 's'}` : `${name}s`, 'sim-small'), ...grid.nodes]));
		y += Math.max(grid.height, 14) + 14;
	}
	return {
		height: y + PAD,
		nodes: [box(PAD, 22, w - 2 * PAD, y - 22, 'sim-die'), label(PAD, 15, 'One SM partition'), ...nodes],
	};
}

function memory(g: GpuSpec, w: number, sel: string, on: Select) {
	const levels = memoryLevels(g);
	const maxLog = Math.log10(levels[levels.length - 1].totalKb);
	const barMax = w - 2 * PAD;
	const nodes: SVGElement[] = [label(PAD, 15, 'Whole-GPU capacity, log scale')];
	levels.forEach((m, i) => {
		const y = 30 + i * 58;
		const bw = Math.max(24, (Math.log10(m.totalKb) / maxLog) * barMax);
		nodes.push(
			part(m.id, sel, on, [
				label(PAD, y + 12, `${m.name}: ${formatKb(m.totalKb)}`),
				box(PAD, y + 20, bw, 22, m.id === 'dram' ? 'sim-box is-dram' : 'sim-box is-cache'),
			]),
		);
	});
	return { height: 30 + levels.length * 58 + PAD, nodes };
}

const DRAW: Record<Level, typeof chip> = { chip, sm, partition, memory };

export function render(root: SVGSVGElement, g: GpuSpec, step: number, selected: string, width: number, onSelect: Select) {
	const { height, nodes } = DRAW[levelAt(step)](g, width, selected, onSelect);
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes);
}
