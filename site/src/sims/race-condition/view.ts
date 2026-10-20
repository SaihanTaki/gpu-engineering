import { clear, svg } from '@sim-core/svg';
import { isComplete, OP_TEXT, OPS, schedule, stateAt, type Config } from './model';

const PAD = 12;
const GAP = 8;
const CARD_MIN_W = 172;
const ROW_H = 20;
const CHIP_W = 76;
const CHIP_H = 34;
const BAR_H = 14;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: 3, class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

const OP_SHORT = { load: 'load', add: 'add', store: 'store', atomic: 'atomic' } as const;

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	const events = schedule(c);
	const s = stateAt(c, step);
	const ops = OPS[c.mode];
	const ran = s.event;
	let y = PAD;

	// Shared memory: the one counter every thread updates.
	nodes.push(text(PAD, y + 12, 'shared memory', 'sim-small'));
	y += 18;
	const memW = 132;
	const memH = 54;
	const lostNow = ran?.op === 'store' && ran.lost > 0;
	nodes.push(rect(PAD, y, memW, memH, lostNow ? 'rc-mem is-lost' : ran && ran.after !== ran.before ? 'rc-mem is-now' : 'rc-mem'));
	nodes.push(text(PAD + 8, y + 14, 'count', 'sim-small'));
	nodes.push(text(PAD + 8, y + 45, String(s.count), 'rc-value'));
	const notes = [`expected at the end: ${c.threads}`, `updates lost so far: ${s.lost}`];
	if (lostNow) notes.push(`this store lost ${ran.lost}`);
	const side = inner - memW - GAP >= 190;
	const nx = side ? PAD + memW + 12 : PAD;
	const ny = side ? y + 4 : y + memH + 8;
	notes.forEach((n, i) => nodes.push(text(nx, ny + 12 + i * 16, n, i === 2 ? 'sim-small rc-lost-text' : 'sim-small')));
	y = side ? y + Math.max(memH, notes.length * 16 + 4) : ny + notes.length * 16;
	y += 14;

	// One card per thread: its register and its instructions.
	const perRow = Math.max(1, Math.min(c.threads, Math.floor((inner + GAP) / (CARD_MIN_W + GAP))));
	const cardW = (inner - (perRow - 1) * GAP) / perRow;
	const cardH = 26 + 30 + ops.length * ROW_H + 8;
	for (let t = 0; t < c.threads; t++) {
		const th = s.threads[t];
		const cx = PAD + (t % perRow) * (cardW + GAP);
		const cy = y + Math.floor(t / perRow) * (cardH + GAP);
		const justRan = ran?.thread === t;
		nodes.push(rect(cx, cy, cardW, cardH, `rc-card${t % 2 ? ' is-b' : ''}${justRan ? ' is-now' : ''}`));
		nodes.push(text(cx + 8, cy + 17, `thread ${t}`));
		const state = th.done >= ops.length ? 'finished' : `next: ${OP_SHORT[ops[th.done]]}`;
		nodes.push(text(cx + cardW - 8, cy + 17, state, 'sim-small', 'end'));
		// The register, private to this thread.
		nodes.push(rect(cx + 8, cy + 26, cardW - 16, 24, 'rc-reg'));
		nodes.push(text(cx + 14, cy + 42, `r = ${th.reg === null ? '?' : th.reg}`, 'sim-label'));
		ops.forEach((op, i) => {
			const ry = cy + 56 + i * ROW_H;
			const isRan = justRan && i === th.done - 1;
			if (isRan) nodes.push(rect(cx + 4, ry, cardW - 8, ROW_H - 2, 'rc-ran'));
			if (i === th.done) nodes.push(svg('path', { d: `M${cx + 9} ${ry + 4} l7 5 l-7 5 z`, class: 'rc-next' }));
			nodes.push(text(cx + 22, ry + 13, OP_TEXT[op], i < th.done ? 'sim-small' : 'sim-small rc-op'));
		});
	}
	y += Math.ceil(c.threads / perRow) * (cardH + GAP) + 6;

	// The interleaving: every instruction in the order it runs.
	nodes.push(text(PAD, y + 12, 'order of instructions'));
	y += 20;
	if (events.length === 0) {
		nodes.push(text(PAD, y + 12, 'none yet: pick a thread to run', 'sim-small'));
		y += 22;
	} else {
		const perLine = Math.max(1, Math.floor((inner + 4) / (CHIP_W + 4)));
		events.forEach((e, i) => {
			const cx = PAD + (i % perLine) * (CHIP_W + 4);
			const cy = y + Math.floor(i / perLine) * (CHIP_H + 4);
			const past = i < step;
			const lossy = past && s.lossySteps.includes(i);
			const cls = !past ? 'rc-chip is-future' : lossy ? 'rc-chip is-lost' : i === step - 1 ? 'rc-chip is-now' : 'rc-chip';
			nodes.push(rect(cx, cy, CHIP_W, CHIP_H, cls));
			nodes.push(text(cx + 6, cy + 14, `T${e.thread} ${OP_SHORT[e.op]}`, past ? 'sim-small rc-op' : 'sim-small'));
			if (past) {
				const at = stateAt(c, i + 1);
				const r = at.threads[e.thread].reg;
				const info = lossy ? `lost ${at.event!.lost}` : e.op === 'store' || e.op === 'atomic' ? `count=${at.count}` : `r=${r}`;
				nodes.push(text(cx + 6, cy + 28, info, lossy ? 'sim-small rc-lost-text' : 'sim-small'));
			}
		});
		y += Math.ceil(events.length / perLine) * (CHIP_H + 4);
	}
	y += 10;

	// Expected against actual, on one scale.
	const done = step === events.length && isComplete(c);
	nodes.push(text(PAD, y + 12, done ? 'result' : 'result (so far)'));
	y += 20;
	const bars: [string, number, string][] = [
		[`expected: ${c.threads}`, c.threads, 'rc-bar'],
		[`count: ${s.count}`, s.count, done && s.lost > 0 ? 'rc-bar is-lost' : 'rc-bar'],
	];
	for (const [label, v, cls] of bars) {
		nodes.push(text(PAD, y + 11, label, 'sim-small rc-op'));
		y += 16;
		nodes.push(rect(PAD, y, inner, BAR_H, 'rc-bar-track'));
		if (v > 0) nodes.push(rect(PAD, y, (inner * v) / c.threads, BAR_H, cls));
		y += BAR_H + 8;
	}
	y += PAD - 8;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
