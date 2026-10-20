import { clear, svg } from '@sim-core/svg';
import type { GpuSpec } from '@sim-core/gpu-specs';
import { arrayBytes, blocksFor, fmt, fmtBytes, fmtTime, times, type BufId, type BufState, type State } from './model';

const PAD = 12;
const SLOT_H = 38;
const SM = 24;
const SM_GAP = 6;
const BAR_H = 18;

const NAMES: Record<BufId, string> = { hA: 'h_a', hB: 'h_b', hC: 'h_c', dA: 'd_a', dB: 'd_b', dC: 'd_c' };

const text = (x: number, y: number, s: string, cls = 'sim-label') => svg('text', { x, y, class: cls }, [s]);

function arrow(x1: number, y1: number, x2: number, y2: number) {
	const dir = Math.sign(y2 - y1);
	return [
		svg('line', { x1, y1, x2, y2: y2 - dir * 6, class: 'dj-arrow' }),
		svg('polygon', { points: `${x2 - 5},${y2 - dir * 7} ${x2 + 5},${y2 - dir * 7} ${x2},${y2}`, class: 'dj-arrowhead' }),
	];
}

/** Three buffer slots in a row; returns the nodes and each slot's center x. */
function slots(ids: BufId[], st: State, n: number, x: number, y: number, w: number) {
	const gap = 8;
	const sw = (w - 2 * gap) / 3;
	const centers: Partial<Record<BufId, number>> = {};
	const nodes: SVGElement[] = [];
	ids.forEach((id, i) => {
		const sx = x + i * (sw + gap);
		const state: BufState = st.buffers[id];
		const moving = st.transfer && (st.transfer.from === id || st.transfer.to === id);
		centers[id] = sx + sw / 2;
		nodes.push(
			svg('rect', { x: sx, y, width: sw, height: SLOT_H, rx: 3, class: `dj-buf is-${state}${moving ? ' is-moving' : ''}` }),
			text(sx + 8, y + 16, NAMES[id], state === 'none' ? 'sim-small' : 'sim-label'),
			text(sx + 8, y + 31, state === 'full' ? fmtBytes(arrayBytes(n)) : state === 'empty' ? 'empty' : 'free', 'sim-small'),
		);
	});
	return { nodes, centers };
}

export function render(root: SVGSVGElement, g: GpuSpec, n: number, st: State, width: number) {
	const inner = width - 2 * PAD;
	const out: SVGElement[] = [
		svg('defs', {}, [
			svg('pattern', { id: 'dj-stripes', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 5, height: 5, class: 'dj-stripe-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, class: 'dj-stripe' }),
			]),
		]),
	];

	// CPU RAM
	out.push(text(PAD, 15, 'CPU RAM (host)'), svg('rect', { x: PAD, y: 22, width: inner, height: SLOT_H + 20, rx: 3, class: 'sim-die' }));
	const host = slots(['hA', 'hB', 'hC'], st, n, PAD + 10, 32, inner - 20);
	out.push(...host.nodes);

	// PCIe link
	const linkTop = 22 + SLOT_H + 20;
	const linkH = 80;
	const lx = PAD + 18;
	out.push(svg('rect', { x: lx - 5, y: linkTop, width: 10, height: linkH, class: `dj-link${st.transfer ? ' is-busy' : ''}` }));
	out.push(text(lx + 22, linkTop + 20, `PCIe Gen${g.pcieGen} x16`), text(lx + 22, linkTop + 36, `${g.pcieGbPerS} GB/s each way, peak`, 'sim-small'));
	if (st.transfer) {
		const down = st.transfer.to.startsWith('d');
		out.push(...arrow(lx, down ? linkTop + 4 : linkTop + linkH - 4, lx, down ? linkTop + linkH - 4 : linkTop + 4));
		const t = times(g, n);
		out.push(
			text(lx + 22, linkTop + 54, `${NAMES[st.transfer.from]} to ${NAMES[st.transfer.to]}, ${fmtBytes(arrayBytes(n))}`, 'sim-small'),
			text(lx + 22, linkTop + 69, `at least ${fmtTime(down ? t.copyIn / 2 : t.copyOut)}`, 'sim-small'),
		);
	} else {
		out.push(text(lx + 22, linkTop + 54, 'idle', 'sim-small'));
	}

	// GPU: HBM and a row of SMs
	const gy = linkTop + linkH;
	const hbmY = gy + 28;
	const hbmH = SLOT_H + 32;
	const smY = hbmY + hbmH + 40;
	const gpuH = smY + SM + 44 - gy;
	out.push(
		svg('rect', { x: PAD, y: gy, width: inner, height: gpuH, rx: 3, class: 'sim-die' }),
		text(PAD + 10, gy + 18, `GPU: ${g.name}`),
		svg('rect', { x: PAD + 10, y: hbmY, width: inner - 20, height: hbmH, rx: 3, class: 'sim-box is-dram' }),
		text(PAD + 18, hbmY + 16, `${g.memory}, ${fmt(g.memoryGbPerS)} GB/s`, 'sim-small'),
	);
	const dev = slots(['dA', 'dB', 'dC'], st, n, PAD + 18, hbmY + 24, inner - 36);
	out.push(...dev.nodes);

	const drawn = Math.max(1, Math.min(g.sms, Math.floor((inner - 20 + SM_GAP) / (SM + SM_GAP))));
	const busy = Math.min(blocksFor(n), g.sms);
	const active = st.sm === 'assigned' || st.sm === 'read' || st.sm === 'write';
	for (let i = 0; i < drawn; i++) {
		const on = active && i < busy;
		out.push(svg('rect', { x: PAD + 10 + i * (SM + SM_GAP), y: smY, width: SM, height: SM, rx: 2, class: `dj-sm${on ? ' is-busy' : ''}` }));
	}
	if (st.sm === 'read') {
		for (const id of ['dA', 'dB'] as const) out.push(...arrow(dev.centers[id]!, hbmY + 24 + SLOT_H + 2, dev.centers[id]!, smY - 2));
	} else if (st.sm === 'write') {
		out.push(...arrow(dev.centers.dC!, smY - 2, dev.centers.dC!, hbmY + 24 + SLOT_H + 2));
	}
	const smText = {
		idle: 'idle',
		assigned: `${fmt(blocksFor(n))} blocks to run`,
		read: `reading a[i], b[i]: ${fmtBytes(2 * arrayBytes(n))}`,
		write: `writing c[i]: ${fmtBytes(arrayBytes(n))}`,
		done: 'kernel finished',
	}[st.sm];
	out.push(
		text(PAD + 10, smY + SM + 16, `${g.sms} SMs (${drawn} drawn)`, 'sim-small'),
		text(PAD + 10, smY + SM + 31, smText, 'sim-small'),
	);

	// Time bar
	const t = times(g, n);
	const by = gy + gpuH + 36;
	const elapsed = st.done.copyIn + st.done.kernel + st.done.copyOut;
	out.push(text(PAD, by - 8, `Time: ${fmtTime(elapsed)} of ${fmtTime(t.total)}`));
	// Keep the kernel segment visible even when it is a sliver of the total.
	const minW = 4;
	const kernelW = Math.max(minW, (inner * t.kernel) / t.total);
	const scale = (inner - kernelW) / (t.copyIn + t.copyOut);
	const segs = [
		{ w: t.copyIn * scale, done: st.done.copyIn / t.copyIn, cls: 'dj-seg-copy' },
		{ w: kernelW, done: st.done.kernel / t.kernel, cls: 'dj-seg-kernel' },
		{ w: t.copyOut * scale, done: st.done.copyOut / t.copyOut, cls: 'dj-seg-copy' },
	];
	let x = PAD;
	for (const s of segs) {
		out.push(
			svg('rect', { x, y: by, width: s.w, height: BAR_H, class: 'dj-seg-track' }),
			svg('rect', { x, y: by, width: s.w * s.done, height: BAR_H, class: s.cls }),
		);
		x += s.w;
	}
	out.push(svg('rect', { x: PAD, y: by, width: inner, height: BAR_H, class: 'dj-seg-frame' }));

	const height = by + BAR_H + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
