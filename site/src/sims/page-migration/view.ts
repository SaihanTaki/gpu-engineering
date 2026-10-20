import { clear, svg } from '@sim-core/svg';
import type { GpuSpec } from '@sim-core/gpu-specs';
import { fmtMiB, fmtTime, PAGE_BYTES, total, type Cell, type Cost, type Mode, type State } from './model';

const PAD = 12;
const CELL_H = 30;
const GAP = 4;
const BAR_H = 18;

const text = (x: number, y: number, s: string, cls = 'sim-label') => svg('text', { x, y, class: cls }, [s]);

/** Bar segments in drawing order; the class picks the fill. */
export const KINDS: { key: keyof Cost; cls: string; name: string }[] = [
	{ key: 'pcie', cls: 'pm-seg-pcie', name: 'PCIe transfers (peak)' },
	{ key: 'fault', cls: 'pm-seg-fault', name: 'Fault handling (assumed)' },
	{ key: 'staging', cls: 'pm-seg-staging', name: 'Staging copies (assumed)' },
	{ key: 'kernel', cls: 'pm-seg-kernel', name: 'Kernel' },
];

function grid(cells: Cell[], st: State, side: 'cpu' | 'gpu', x: number, y: number, w: number) {
	const cols = Math.min(cells.length, 8);
	const cw = (w - (cols - 1) * GAP) / cols;
	const moving = new Set(st.transfer ? st.transfer.pages : []);
	const out: SVGElement[] = [];
	cells.forEach((c, k) => {
		const cx = x + (k % cols) * (cw + GAP);
		const cy = y + Math.floor(k / cols) * (CELL_H + GAP);
		const faulted = st.fault && st.fault.side === side && st.fault.page === k;
		// The page is drawn as moving on the side it arrives at.
		const arrived = moving.has(k) && c !== 'none';
		out.push(svg('rect', { x: cx, y: cy, width: cw, height: CELL_H, rx: 2, class: `pm-page is-${c}${arrived ? ' is-moving' : ''}${faulted ? ' is-fault' : ''}` }));
		if (c === 'x' || c === '2x') out.push(svg('text', { x: cx + cw / 2, y: cy + 19, class: 'pm-page-text' }, [c]));
	});
	const rows = Math.ceil(cells.length / cols);
	return { nodes: out, height: rows * CELL_H + (rows - 1) * GAP };
}

function arrow(x: number, y1: number, y2: number) {
	const dir = Math.sign(y2 - y1);
	return [
		svg('line', { x1: x, y1, x2: x, y2: y2 - dir * 6, class: 'pm-arrow' }),
		svg('polygon', { points: `${x - 5},${y2 - dir * 7} ${x + 5},${y2 - dir * 7} ${x},${y2}`, class: 'pm-arrowhead' }),
	];
}

const residentLabel = (mode: Mode, side: 'cpu' | 'gpu') => {
	if (mode === 'demand' || mode === 'prefetch') return side === 'cpu' ? 'x: pages resident on the CPU' : 'x: pages resident on the GPU';
	if (side === 'gpu') return 'd_x (cudaMalloc)';
	return mode === 'pinned' ? 'h_x, pinned (cudaMallocHost)' : 'h_x, pageable (malloc)';
};

export function render(root: SVGSVGElement, g: GpuSpec, mode: Mode, st: State, end: Cost, width: number) {
	const inner = width - 2 * PAD;
	const out: SVGElement[] = [
		svg('defs', {}, [
			svg('pattern', { id: 'pm-stripes', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 5, height: 5, class: 'pm-stripe-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'pm-stripe' }),
			]),
		]),
	];

	// CPU RAM
	const cpuGrid = grid(st.cpu, st, 'cpu', PAD + 10, 48, inner - 20);
	const staging = mode === 'pageable';
	const stageY = 48 + cpuGrid.height + 10;
	const cpuH = 48 + cpuGrid.height + (staging ? CELL_H + 30 : 12) - 22;
	out.push(
		text(PAD, 15, 'CPU RAM (host)'),
		svg('rect', { x: PAD, y: 22, width: inner, height: cpuH, rx: 3, class: 'sim-die' }),
		text(PAD + 10, 40, residentLabel(mode, 'cpu'), 'sim-small'),
		...cpuGrid.nodes,
	);
	if (staging) {
		const busy = st.staging === 'x' || st.staging === '2x';
		out.push(
			svg('rect', { x: PAD + 10, y: stageY + 8, width: inner - 20, height: CELL_H, rx: 2, class: `pm-page is-${st.staging === 'none' ? 'none' : st.staging}${busy ? ' is-moving' : ''}` }),
			text(PAD + 18, stageY + 27, busy ? `staging buffer (pinned): ${st.staging}` : 'driver staging buffer (pinned)', busy ? 'pm-page-label' : 'sim-small'),
		);
	}

	// PCIe link
	const linkTop = 22 + cpuH;
	const linkH = 78;
	const lx = PAD + 18;
	out.push(svg('rect', { x: lx - 5, y: linkTop, width: 10, height: linkH, class: `pm-link${st.transfer ? ' is-busy' : ''}` }));
	out.push(text(lx + 22, linkTop + 20, `PCIe Gen${g.pcieGen} x16`), text(lx + 22, linkTop + 36, `${g.pcieGbPerS} GB/s each way, peak`, 'sim-small'));
	if (st.transfer) {
		const down = st.transfer.dir === 'h2d';
		const n = st.transfer.pages.length;
		out.push(...arrow(lx, down ? linkTop + 4 : linkTop + linkH - 4, down ? linkTop + linkH - 4 : linkTop + 4));
		out.push(
			text(lx + 22, linkTop + 54, `${down ? 'to GPU' : 'to CPU'}: ${n === 1 ? `page ${st.transfer.pages[0]}` : `${n} pages`}, ${fmtMiB(n * PAGE_BYTES)}`, 'sim-small'),
			text(lx + 22, linkTop + 69, st.fault ? `${st.fault.side.toUpperCase()} page fault` : 'bulk transfer', st.fault ? 'pm-fault-text' : 'sim-small'),
		);
	} else {
		out.push(text(lx + 22, linkTop + 54, 'idle', 'sim-small'));
	}

	// GPU memory
	const gy = linkTop + linkH;
	const gpuGrid = grid(st.gpu, st, 'gpu', PAD + 10, gy + 44, inner - 20);
	const gpuH = 44 + gpuGrid.height + 12;
	out.push(
		svg('rect', { x: PAD, y: gy, width: inner, height: gpuH, rx: 3, class: 'sim-die' }),
		text(PAD + 10, gy + 18, `GPU: ${g.name}, ${g.memory}`),
		text(PAD + 10, gy + 34, residentLabel(mode, 'gpu'), 'sim-small'),
		...gpuGrid.nodes,
	);

	// Time bar, one segment per kind, scaled to the whole run.
	const by = gy + gpuH + 36;
	const runTotal = total(end);
	out.push(text(PAD, by - 8, `Time: ${fmtTime(total(st.done))} of ${fmtTime(runTotal)}`));
	const shown = KINDS.filter((k) => end[k.key] > 0);
	// Keep small segments visible.
	const minW = 4;
	const widths = shown.map((k) => Math.max(minW, (inner * end[k.key]) / runTotal));
	const scale = inner / widths.reduce((a, b) => a + b, 0);
	let x = PAD;
	shown.forEach((k, i) => {
		const w = widths[i] * scale;
		out.push(
			svg('rect', { x, y: by, width: w, height: BAR_H, class: 'pm-seg-track' }),
			svg('rect', { x, y: by, width: w * (st.done[k.key] / end[k.key]), height: BAR_H, class: k.cls }),
		);
		x += w;
	});
	out.push(svg('rect', { x: PAD, y: by, width: inner, height: BAR_H, class: 'pm-seg-frame' }));

	const height = by + BAR_H + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
