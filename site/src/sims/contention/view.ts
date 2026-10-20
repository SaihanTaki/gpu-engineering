import { clear, svg } from '@sim-core/svg';
import {
	BINS,
	globalHistAt,
	globalQueues,
	KERNEL_NAMES,
	LAST_STEP,
	MERGE_STEP,
	mergeQueues,
	privateCounts,
	ROUNDS,
	sharedQueues,
	stepSlots,
	threadsPerBlock,
	totalsAt,
	type Config,
	type Queue,
} from './model';

const PAD = 12;
/** Height of one queued atomic, the same in every panel so queues compare by eye. */
const UNIT = 4;
const COL_GAP = 3;
const GROUP_GAP = 10;
const MIN_GROUP_W = 8 * 22;
const BAR_H = 16;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

const isRound = (step: number) => step >= 1 && step <= ROUNDS;
const longest = (qs: Queue[]) => Math.max(0, ...qs.map((q) => q.length));

/** Longest queue any round of this panel reaches, so its height does not jump between steps. */
function peak(c: Config, kind: 'global' | 'shared' | 'merge') {
	if (kind === 'merge') return longest(mergeQueues(c));
	let m = 0;
	for (let r = 1; r <= ROUNDS; r++) m = Math.max(m, kind === 'global' ? longest(globalQueues(c, r)) : longest(sharedQueues(c, r).flat()));
	return m;
}

/**
 * BINS columns of queued atomics standing on a baseline, with the bin number
 * and a count below. Returns the y below the drawing.
 */
function drawBins(nodes: SVGElement[], x: number, top: number, w: number, queues: Queue[] | null, counts: number[], height: number) {
	const colW = (w - (BINS - 1) * COL_GAP) / BINS;
	const base = top + 14 + height;
	const worst = queues ? longest(queues) : 0;
	for (let b = 0; b < BINS; b++) {
		const cx = x + b * (colW + COL_GAP);
		const q = queues?.[b] ?? [];
		// The longest queue sets the step's cost, so it is marked in the warning color.
		const cls = q.length === worst && worst > 1 ? 'ct-atomic is-worst' : 'ct-atomic';
		q.forEach((_, i) => nodes.push(rect(cx, base - (i + 1) * UNIT, colW, UNIT - 1, cls)));
		// The length in text too, so the longest queue does not rely on color alone.
		if (q.length === worst && worst > 1) nodes.push(text(cx + colW / 2, base - q.length * UNIT - 3, String(worst), 'sim-small ct-costly', 'middle'));
		nodes.push(svg('line', { x1: cx, x2: cx + colW, y1: base + 0.5, y2: base + 0.5, class: 'ct-base' }));
		nodes.push(text(cx + colW / 2, base + 13, String(b), 'sim-small', 'middle'));
		nodes.push(text(cx + colW / 2, base + 27, String(counts[b]), 'sim-small ct-count', 'middle'));
	}
	return base + 32;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	const tpb = threadsPerBlock(c);
	let y = PAD;

	// Global kernel.
	const gq = isRound(step) ? globalQueues(c, step) : null;
	nodes.push(text(PAD, y + 12, KERNEL_NAMES.global));
	const gNote = step === 0 ? 'not started' : gq ? `round ${step}: longest queue ${longest(gq)}` : 'done';
	nodes.push(text(PAD, y + 28, gNote, gq && longest(gq) > 4 ? 'sim-small ct-costly' : 'sim-small'));
	y = drawBins(nodes, PAD, y + 38, inner, gq, globalHistAt(c, 'global', step), Math.max(8, peak(c, 'global')) * UNIT);
	nodes.push(svg('line', { x1: PAD, x2: PAD + inner, y1: y, y2: y, class: 'ct-rule' }));
	y += 8;

	// Shared kernel: one private copy per block, then the merge into global memory.
	const sq = isRound(step) ? sharedQueues(c, step) : null;
	nodes.push(text(PAD, y + 12, KERNEL_NAMES.shared));
	const sNote =
		step === 0 ? 'not started' : sq ? `round ${step}: longest queue ${stepSlots(c, 'shared', step)}` : step === MERGE_STEP ? `merge: longest queue ${stepSlots(c, 'shared', step)}` : 'done';
	nodes.push(text(PAD, y + 28, sNote, sq && stepSlots(c, 'shared', step) > 4 ? 'sim-small ct-costly' : 'sim-small'));
	y += 38;

	const perRow = Math.max(1, Math.min(c.blocks, Math.floor((inner + GROUP_GAP) / (MIN_GROUP_W + GROUP_GAP))));
	const groupW = (inner - (perRow - 1) * GROUP_GAP) / perRow;
	const sHeight = Math.max(8, peak(c, 'shared')) * UNIT;
	const counts = privateCounts(c, step);
	const groupH = 20 + 14 + sHeight + 32 + 6;
	for (let blk = 0; blk < c.blocks; blk++) {
		const gx = PAD + (blk % perRow) * (groupW + GROUP_GAP);
		const gy = y + Math.floor(blk / perRow) * (groupH + GROUP_GAP);
		nodes.push(rect(gx, gy, groupW, groupH, blk % 2 ? 'ct-group is-b' : 'ct-group'));
		nodes.push(text(gx + 6, gy + 14, `block ${blk}, ${tpb} threads`, 'sim-small ct-group-label'));
		drawBins(nodes, gx + 6, gy + 20, groupW - 12, sq ? sq[blk] : null, counts[blk], sHeight);
	}
	y += Math.ceil(c.blocks / perRow) * (groupH + GROUP_GAP);

	const mq = step === MERGE_STEP ? mergeQueues(c) : null;
	nodes.push(text(PAD, y + 12, 'merge into global memory', 'sim-small'));
	y = drawBins(nodes, PAD, y + 20, inner, mq, globalHistAt(c, 'shared', step), Math.max(8, peak(c, 'merge')) * UNIT);
	nodes.push(svg('line', { x1: PAD, x2: PAD + inner, y1: y, y2: y, class: 'ct-rule' }));
	y += 8;

	// Serialized slots so far, on one scale for both kernels.
	nodes.push(text(PAD, y + 12, 'Serialized slots (model)'));
	nodes.push(text(PAD, y + 28, 'sum of each step\'s longest queue', 'sim-small'));
	y += 40;
	const scale = Math.max(totalsAt(c, 'global', LAST_STEP).slots, totalsAt(c, 'shared', LAST_STEP).slots);
	for (const k of ['global', 'shared'] as const) {
		const slots = totalsAt(c, k, step).slots;
		nodes.push(text(PAD, y + 12, `${k === 'global' ? 'Global' : 'Shared'}: ${slots}`, 'sim-small ct-bar-label'));
		y += 18;
		nodes.push(rect(PAD, y, inner, BAR_H, 'ct-bar-track'));
		if (slots > 0) nodes.push(rect(PAD, y, (inner * slots) / scale, BAR_H, 'ct-bar'));
		y += BAR_H + 10;
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
