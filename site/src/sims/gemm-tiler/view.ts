import { clear, svg } from '@sim-core/svg';
import { gpuById } from '@sim-core/gpu-specs';
import {
	analyzeStep,
	fmt,
	LAST_STEP,
	LIMIT_NAMES,
	NAIVE_BLOCK,
	peakGflops,
	STEP_IDS,
	STEP_NAMES,
	stepAt,
	WARP,
	type Analysis,
	type Config,
	type Limit,
} from './model';

const PAD = 12;
const GAP = 14;
const BAR_H = 14;

const rect = (x: number, y: number, w: number, h: number, cls: string, extra: Record<string, string | number> = {}) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), class: cls, ...extra });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/** Heat 0..1 on a log scale: 1 load per element is cold, `size` loads (every thread its own) is hot. */
const heat = (loads: number, size: number) => Math.log(Math.max(1, loads)) / Math.log(size);
const heatClass = (h: number) => `gt-heat gt-heat-${Math.min(4, Math.round(h * 4))}`;

/** A, B and C in the usual layout: B above C, A to the left. Block 0 (or warp 0) and its slices outlined. */
function drawMatrices(nodes: SVGElement[], c: Config, a: Analysis, top: number, inner: number) {
	const S = Math.floor(Math.min(170, (inner - GAP - 18) / 2));
	const left = PAD + 18;
	const ax = left, cx = left + S + GAP, by = top + 16, cy = by + S + GAP;
	const sc = S / c.size;
	const px = (n: number) => Math.max(2, n * sc);
	const naive = a.step === 'naive';
	const coalesced = a.step === 'coalesced';
	const t = a.tile;
	// Footprint of what one warp (naive steps) or one block (tiled steps) works on, in elements.
	const rows = naive ? WARP : coalesced ? 1 : t.bm;
	const cols = naive ? 1 : coalesced ? WARP : t.bn;
	const kStep = naive || coalesced ? 1 : t.bk;

	nodes.push(text(cx, top + 10, `B  ${fmt(c.size)} x ${fmt(c.size)}`, 'sim-small'));
	nodes.push(text(ax, cy - 4, 'A', 'sim-small'));
	nodes.push(text(cx, cy + S + 14, 'C', 'sim-small'));

	const hA = heat(a.reuse.a, c.size), hB = heat(a.reuse.b, c.size);
	nodes.push(rect(ax, cy, S, S, heatClass(hA)), rect(cx, by, S, S, heatClass(hB)), rect(cx, cy, S, S, heatClass(0)));
	// Block grid of C, at most 16 lines a side so it stays readable.
	if (!naive && !coalesced) {
		const nr = Math.min(16, c.size / t.bm), nc = Math.min(16, c.size / t.bn);
		for (let i = 1; i < nr; i++) nodes.push(svg('line', { x1: cx, x2: cx + S, y1: cy + (S * i) / nr, y2: cy + (S * i) / nr, class: 'gt-grid' }));
		for (let j = 1; j < nc; j++) nodes.push(svg('line', { x1: cx + (S * j) / nc, x2: cx + (S * j) / nc, y1: cy, y2: cy + S, class: 'gt-grid' }));
	}
	for (const [x, y] of [[ax, cy], [cx, by], [cx, cy]]) nodes.push(rect(x, y, S, S, 'gt-frame'));

	// A: the block's rows over all of K, the current k-tile stronger. B: the block's columns, likewise.
	nodes.push(rect(ax, cy, S, px(rows), 'gt-strip'), rect(ax, cy, px(kStep), px(rows), 'gt-slice'));
	nodes.push(rect(cx, by, px(cols), S, 'gt-strip'), rect(cx, by, px(cols), px(kStep), 'gt-slice'));
	nodes.push(rect(cx, cy, px(cols), px(rows), 'gt-slice'));

	const who = naive || coalesced ? 'warp 0' : 'block 0';
	nodes.push(text(PAD, cy + S + 30, `outlined: ${who}'s data`, 'sim-small'));
	return cy + S + 36;
}

/** Heat legend: global loads per element of A and B. */
function drawLegend(nodes: SVGElement[], c: Config, a: Analysis, top: number, inner: number) {
	const w = Math.min(inner, 260);
	nodes.push(text(PAD, top + 11, 'Global loads per element (log)', 'sim-small'));
	const y = top + 18;
	for (let i = 0; i <= 4; i++) nodes.push(rect(PAD + (w * i) / 5, y, w / 5, 10, `gt-heat gt-heat-${i}`));
	nodes.push(rect(PAD, y, w, 10, 'gt-frame'));
	nodes.push(text(PAD, y + 24, '1', 'sim-small'), text(PAD + w, y + 24, fmt(c.size), 'sim-small', 'end'));
	nodes.push(text(PAD, y + 40, `A ${fmt(a.reuse.a)}x, B ${fmt(a.reuse.b)}x, C 1x`, 'sim-small gt-strong'));
	return y + 50;
}

/** One cell per thread of the block (each owns TM x TN outputs); warp 0 filled, thread 0 outlined. */
function drawBlock(nodes: SVGElement[], a: Analysis, top: number, inner: number) {
	const naive = a.step === 'naive' || a.step === 'coalesced';
	const t = a.tile;
	const tr = naive ? NAIVE_BLOCK : t.bm / t.tm;
	const tc = naive ? NAIVE_BLOCK : t.bn / t.tn;
	nodes.push(text(PAD, top + 12, `Block: ${tr} x ${tc} threads`, 'sim-small gt-strong'));
	nodes.push(text(PAD, top + 26, naive ? '1 output per thread' : `${t.tm} x ${t.tn} outputs per thread`, 'sim-small'));
	const y0 = top + 34;
	const maxW = Math.min(inner, 240);
	const cell = Math.max(1, Math.min(14, Math.floor(maxW / tc), Math.floor(160 / tr)));
	// Threads in a warp: in the naive kernel threadIdx.x walks rows, elsewhere it walks columns.
	const inWarp = (r: number, col: number) => (a.step === 'naive' ? col === 0 && r < WARP : r * tc + col < WARP);
	const dense = cell < 4;
	if (!dense) {
		for (let r = 0; r < tr; r++)
			for (let col = 0; col < tc; col++) nodes.push(rect(PAD + col * cell, y0 + r * cell, cell - 1, cell - 1, inWarp(r, col) ? 'gt-thread is-warp' : 'gt-thread'));
	} else {
		nodes.push(rect(PAD, y0, tc * cell, tr * cell, 'gt-thread'));
		const wr = a.step === 'naive' ? Math.min(WARP, tr) : Math.ceil(WARP / tc);
		const wc = a.step === 'naive' ? 1 : Math.min(WARP, tc);
		nodes.push(rect(PAD, y0, Math.max(2, wc * cell), Math.max(2, wr * cell), 'gt-thread is-warp'));
	}
	nodes.push(rect(PAD, y0, Math.max(3, cell - 1), Math.max(3, cell - 1), 'gt-me'));
	const bottom = y0 + tr * cell;
	nodes.push(text(PAD, bottom + 14, 'filled: warp 0', 'sim-small'), text(PAD, bottom + 28, 'outlined: thread 0', 'sim-small'));
	return bottom + 36;
}

const LOG_MIN = 1, LOG_MAX = 6;
const logX = (v: number, w: number) => (w * (Math.min(LOG_MAX, Math.max(LOG_MIN, Math.log10(v))) - LOG_MIN)) / (LOG_MAX - LOG_MIN);

/** The three ceilings for the current step on a log axis; the lowest is the estimate. */
function drawRoofs(nodes: SVGElement[], a: Analysis, top: number, inner: number) {
	nodes.push(text(PAD, top + 12, 'Model ceilings, GFLOP/s (log)'));
	let y = top + 22;
	for (const k of ['compute', 'dram', 'shared'] as Limit[]) {
		const v = a.roofs[k];
		const finite = Number.isFinite(v);
		const label = `${LIMIT_NAMES[k]}: ${finite ? fmt(v) : 'none'}`;
		nodes.push(text(PAD, y + 10, label, k === a.limit ? 'sim-small gt-strong' : 'sim-small'));
		y += 14;
		nodes.push(rect(PAD, y, inner, BAR_H, 'gt-track'));
		if (finite) nodes.push(rect(PAD, y, logX(v, inner), BAR_H, k === a.limit ? 'gt-bar' : 'gt-bar is-dim'));
		y += BAR_H + 6;
	}
	const TICKS = ['10', '100', '1k', '10k', '100k', '1M'];
	// Narrow sims label every other decade so the ticks do not run together.
	const every = inner < 360 ? 2 : 1;
	for (let e = LOG_MIN; e <= LOG_MAX; e += every) {
		const x = PAD + logX(10 ** e, inner);
		const anchor = e === LOG_MIN ? 'start' : e === LOG_MAX ? 'end' : 'middle';
		nodes.push(text(x, y + 8, TICKS[e - LOG_MIN], 'sim-small', anchor));
	}
	return y + 16;
}

/** Estimate per step against peak, revealed up to the current step so the reader guesses first. */
function drawSteps(nodes: SVGElement[], c: Config, step: number, top: number, inner: number) {
	const g = gpuById(c.gpu);
	const peak = peakGflops(g);
	nodes.push(text(PAD, top + 12, 'Model estimate by step'));
	nodes.push(text(PAD, top + 28, `not measured; peak ${fmt(peak)}`, 'sim-small'));
	let y = top + 36;
	STEP_IDS.forEach((s, i) => {
		const shown = i <= step;
		const a = shown ? analyzeStep(c, s) : null;
		const value = !a ? '?' : a.gflops === null ? 'cannot run' : `${fmt(a.gflops)}`;
		nodes.push(text(PAD, y + 10, `${i + 1}. ${STEP_NAMES[s]}: ${value}`, i === step ? 'sim-small gt-strong' : 'sim-small'));
		y += 14;
		nodes.push(rect(PAD, y, inner, BAR_H, 'gt-track'));
		if (a?.gflops) nodes.push(rect(PAD, y, Math.max(1, (inner * a.gflops) / peak), BAR_H, i === step ? 'gt-bar' : 'gt-bar is-dim'));
		if (a && a.gflops === null) nodes.push(rect(PAD, y, inner, BAR_H, 'gt-track is-error'));
		y += BAR_H + 6;
	});
	return y;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const a = analyzeStep(c, stepAt(step));
	const nodes: SVGElement[] = [];
	let y = PAD;
	y = drawMatrices(nodes, c, a, y, inner);
	y = drawLegend(nodes, c, a, y + 4, inner);
	y = drawBlock(nodes, a, y + 4, inner);
	y = drawRoofs(nodes, a, y + 8, inner);
	y = drawSteps(nodes, c, Math.min(step, LAST_STEP), y + 8, inner);
	y += PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
