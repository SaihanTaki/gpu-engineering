import { clear, svg } from '@sim-core/svg';
import { gpuById } from '@sim-core/gpu-specs';
import {
	AI_MAX,
	AI_MIN,
	attainable,
	bwTbPerS,
	fmtNum,
	kernelPoint,
	peakTflops,
	presetAi,
	presetAt,
	PRESETS,
	ridge,
	ROOF_NAMES,
	roofsOf,
	TF_MAX,
	TF_MIN,
	type Config,
	type RoofId,
} from './model';

const M = { left: 46, right: 10, top: 14, bottom: 42 };
/** Width of one 11px monospace character, for placing labels. */
const CHAR_W = 6.7;

type Box = [number, number, number, number];

const SHORT: Record<RoofId, string> = { fp64: 'FP64', fp32: 'FP32', tf32: 'TF32', fp16: 'FP16', fp8: 'FP8', int8: 'INT8' };

const text = (x: number, y: number, s: string, cls = 'sim-small', anchor?: string, extra: Record<string, string | number> = {}) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor, ...extra }, [s]);
const line = (x1: number, y1: number, x2: number, y2: number, cls: string) => svg('line', { x1, y1, x2, y2, class: cls });

const aiLabel = (ai: number) => (ai < 1 ? `1/${1 / ai}` : ai >= 1024 ? `${ai / 1024}K` : String(ai));

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const g = gpuById(c.gpu);
	const height = Math.round(Math.min(440, Math.max(280, width * 0.68)));
	const pw = width - M.left - M.right;
	const ph = height - M.top - M.bottom;
	const lx = Math.log2(AI_MIN);
	const spanX = Math.log2(AI_MAX) - lx;
	const ly = Math.log10(TF_MIN);
	const spanY = Math.log10(TF_MAX) - ly;
	const X = (ai: number) => M.left + ((Math.log2(ai) - lx) / spanX) * pw;
	const Y = (tf: number) => M.top + ph - ((Math.log10(tf) - ly) / spanY) * ph;
	const clampAi = (ai: number) => Math.min(AI_MAX, Math.max(AI_MIN, ai));
	const clampTf = (tf: number) => Math.min(TF_MAX, Math.max(TF_MIN, tf));
	const nodes: SVGElement[] = [];
	const marks: SVGElement[] = [];
	const boxes: Box[] = [];
	const hits = (b: Box) => boxes.some((o) => b[0] < o[1] && o[0] < b[1] && b[2] < o[3] && o[2] < b[3]);
	const bottom = M.top + ph;
	const right = M.left + pw;
	const narrow = pw < 330;

	// Grid and ticks: every 2 octaves on wide plots, every 4 on narrow ones.
	const every = pw < 420 ? 4 : 2;
	for (let e = lx; e <= lx + spanX; e += every) {
		const x = X(2 ** e);
		nodes.push(line(x, M.top, x, bottom, 'rf-grid'));
		nodes.push(text(x, bottom + 14, aiLabel(2 ** e), 'sim-small', 'middle'));
	}
	for (let e = Math.ceil(ly); e <= ly + spanY; e++) {
		const y = Y(10 ** e);
		nodes.push(line(M.left, y, right, y, 'rf-grid'));
		nodes.push(text(M.left - 4, y + 4, fmtNum(10 ** e), 'sim-small', 'end'));
	}
	nodes.push(svg('rect', { x: M.left, y: M.top, width: pw, height: ph, class: 'rf-frame' }));
	const unit = c.roof === 'int8' ? 'ops' : 'FLOP';
	nodes.push(text(M.left + pw / 2, height - 6, narrow ? `${unit}/byte (log scale)` : `Arithmetic intensity (${unit}/byte, log scale)`, 'sim-small rf-axis', 'middle'));
	nodes.push(
		text(12, M.top + ph / 2, c.roof === 'int8' ? 'TOPS' : 'TFLOPS', 'sim-small rf-axis', 'middle', {
			transform: `rotate(-90 12 ${M.top + ph / 2})`,
		}),
	);

	// Roof lines: one shared bandwidth slope up to the highest ridge, one flat line per precision.
	const roofs = roofsOf(g);
	const bw = bwTbPerS(g);
	const top = Math.max(...roofs.map((r) => peakTflops(g, r)!));
	const selPeak = peakTflops(g, c.roof)!;
	const sel = ridge(g, c.roof);
	nodes.push(line(X(AI_MIN), Y(AI_MIN * bw), X(top / bw), Y(top), 'rf-roof'));
	nodes.push(line(X(AI_MIN), Y(AI_MIN * bw), X(sel), Y(selPeak), 'rf-roof is-selected'));
	// FP8 and INT8 share a value on the H100, so roofs are grouped by value.
	const groups = new Map<number, RoofId[]>();
	for (const r of roofs) {
		const p = peakTflops(g, r)!;
		groups.set(p, [...(groups.get(p) ?? []), r]);
	}
	for (const [p, rs] of groups) nodes.push(line(X(p / bw), Y(p), right, Y(p), `rf-roof${rs.includes(c.roof) ? ' is-selected' : ''}`));

	// Bandwidth label along the slope, just right of the 0.25 FLOP/byte markers.
	const angle = (Math.atan2(Y(AI_MIN * bw) - Y(AI_MIN * 16 * bw), X(AI_MIN * 16) - X(AI_MIN)) * 180) / Math.PI;
	const bx = X(0.25) + 10;
	const by = Y(0.25 * 2 ** ((10 * spanX) / pw) * bw) - 7;
	const bwText = `${narrow ? '' : 'DRAM '}${g.memoryGbPerS.toLocaleString('en-US')} GB/s`;
	nodes.push(text(bx, by, bwText, 'sim-small rf-roof-label', 'start', { transform: `rotate(${-angle} ${bx} ${by})` }));
	const bwLen = bwText.length * CHAR_W;
	const rad = (angle * Math.PI) / 180;
	boxes.push([bx, bx + bwLen * Math.cos(rad) + 9, by - bwLen * Math.sin(rad) - 9, by + 2]);

	// Ridge of the selected roof.
	nodes.push(line(X(sel), Y(selPeak), X(sel), bottom, 'rf-ridge'));
	const leftHalf = X(sel) < M.left + pw * 0.6;
	const ridgeText = `ridge ${fmtNum(sel)}`;
	const rlx = leftHalf ? X(sel) + 4 : X(sel) - 4 - ridgeText.length * CHAR_W;
	nodes.push(text(rlx, bottom - 6, ridgeText, 'sim-small rf-ridge-label', 'start'));
	boxes.push([rlx, rlx + ridgeText.length * CHAR_W, bottom - 15, bottom - 4]);

	// Preset markers, grouped where they share an intensity. Labels go below and
	// right, under the roofline where nothing else is drawn, stacked when close.
	const current = presetAt(step);
	const byAi = new Map<number, number[]>();
	PRESETS.forEach((p, i) => {
		if (p.flops === 0) return;
		const ai = presetAi(p);
		byAi.set(ai, [...(byAi.get(ai) ?? []), i + 1]);
	});
	const flatYs = [...groups.keys()].map(Y);
	let prevEnd = -Infinity;
	let prevY = -Infinity;
	for (const [ai, ids] of [...byAi.entries()].sort((a, b) => a[0] - b[0])) {
		const x = X(clampAi(ai));
		const y = Y(clampTf(attainable(g, c.roof, ai)));
		const on = current !== null && ids.includes(step);
		marks.push(svg('circle', { cx: x, cy: y, r: on ? 7 : 5, class: `rf-preset${on ? ' is-current' : ''}` }));
		boxes.push([x - 7, x + 7, y - 7, y + 7]);
		const label = ids.join(',');
		const lw = label.length * CHAR_W;
		// Below and right first; stack under the previous label when close.
		const below = x + 8 < prevEnd + 2 ? Math.max(y + 16, prevY + 12) : y + 16;
		const spots = [
			{ tx: x + 8, ty: below },
			{ tx: x - 8 - lw, ty: y - 7 },
			{ tx: x - 8 - lw, ty: y + 16 },
			{ tx: x + 8, ty: y - 7 },
		];
		const clean = (sp: { tx: number; ty: number }) =>
			!flatYs.some((fy) => fy > sp.ty - 10 && fy < sp.ty + 3) && !hits([sp.tx, sp.tx + lw, sp.ty - 9, sp.ty + 2]) && sp.tx >= M.left + 2;
		const { tx, ty } = spots.find(clean) ?? spots[0];
		marks.push(text(tx, ty, label, `sim-small rf-preset-label${on ? ' is-current' : ''}`, 'start'));
		prevEnd = tx + lw;
		prevY = ty;
		boxes.push([tx, tx + lw, ty - 9, ty + 2]);
	}

	// Roof labels, selected first. Left of the ridge the chart is empty; right of
	// it a label sits above its own line. Other roofs' labels are dropped when
	// they would collide with something; the selected one always shows.
	const roofYs = [...groups.keys()].map(Y);
	const ordered = [...groups.entries()].sort(([a, ra], [b, rb]) => Number(rb.includes(c.roof)) - Number(ra.includes(c.roof)) || b - a);
	for (const [p, rs] of ordered) {
		const selected = rs.includes(c.roof);
		const label = `${rs.map((r) => (narrow ? SHORT[r] : ROOF_NAMES[r])).join(' / ')} ${fmtNum(p)}`;
		const w = label.length * CHAR_W;
		const rx = X(p / bw);
		const ry = Y(p);
		// Candidates: left of the ridge, then along the flat line above and below it.
		const spots = [{ x1: rx - 6 - w, y: ry + 4 }];
		for (let x1 = rx + 14; x1 <= right - 3 - w; x1 += 10) spots.push({ x1, y: ry - 4 }, { x1, y: ry + 13 });
		spots.push({ x1: right - 3 - w, y: ry - 4 }, { x1: right - 3 - w, y: ry + 13 });
		const inPlot = (sp: { x1: number; y: number }) => sp.x1 >= M.left + 2 && sp.y - 9 >= M.top + 1 && sp.y + 2 <= bottom - 1;
		const fits = spots.filter(inPlot);
		const boxOf = (sp: { x1: number; y: number }): Box => [sp.x1, sp.x1 + w, sp.y - 9, sp.y + 2];
		const crossed = (b: Box) => roofYs.some((oy) => oy !== ry && oy > b[2] && oy < b[3] + 1);
		const spot = fits.find((sp) => !hits(boxOf(sp)) && !crossed(boxOf(sp))) ?? (selected ? (fits[1] ?? fits[0]) : undefined);
		if (!spot) continue;
		boxes.push(boxOf(spot));
		nodes.push(text(spot.x1, spot.y, label, `sim-small rf-roof-label${selected ? ' is-selected' : ''}`, 'start'));
	}

	// The reader's kernel, pinned to the plot edge when it falls outside. Its
	// label goes above and left, away from the preset labels.
	const k = kernelPoint(c);
	const off = k.ai < AI_MIN || k.ai > AI_MAX;
	const kx = X(clampAi(k.ai));
	const ky = Y(clampTf(k.tflops));
	marks.push(line(kx, ky, kx, bottom, 'rf-guide'));
	const d = 8;
	marks.push(svg('path', { d: `M${kx} ${ky - d}L${kx + d} ${ky}L${kx} ${ky + d}L${kx - d} ${ky}Z`, class: 'rf-kernel' }));
	const kLabel = off ? 'your kernel (off chart)' : 'your kernel';
	const kw = kLabel.length * CHAR_W;
	const kx1 = Math.min(right - 3 - kw, Math.max(M.left + 3, kx - 10 - kw));
	marks.push(text(kx1, Math.max(M.top + 11, ky - 10), kLabel, 'sim-small rf-kernel-label', 'start'));

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes, ...marks);
}
