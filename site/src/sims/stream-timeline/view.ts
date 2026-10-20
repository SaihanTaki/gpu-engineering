import { clear, svg } from '@sim-core/svg';
import { schedule, serialParams, type Engine, type Kind, type Op, type Params, type State, fmtTime, makespan } from './model';

const PAD = 10;
const ROW = 26;
const BAR = 18;
const HEAD = 22;

const text = (x: number, y: number, s: string, cls = 'sim-small') => svg('text', { x, y, class: cls }, [s]);

const ENGINE_LABEL: Record<Engine, [string, string]> = {
	h2d: ['H2D engine', 'H2D'],
	compute: ['compute', 'SMs'],
	d2h: ['D2H engine', 'D2H'],
	copy: ['copy engine', 'copy'],
};
const KIND_LETTER: Record<Kind, string> = { h2d: 'H', kernel: 'K', d2h: 'D' };

function bar(o: Op, t: number, y: number, sx: (s: number) => number) {
	const x = sx(o.start);
	const w = Math.max(1.5, sx(o.end) - x);
	const out: SVGElement[] = [];
	const doneW = t >= o.end ? w : t > o.start ? Math.max(1, sx(t) - x) : 0;
	const running = t > o.start && t < o.end;
	out.push(svg('rect', { x, y, width: w, height: BAR, class: 'st-future' }));
	if (doneW > 0) out.push(svg('rect', { x, y, width: doneW, height: BAR, class: `st-${o.kind}` }));
	if (running || (t === o.start && t >= 0)) out.push(svg('rect', { x, y, width: w, height: BAR, class: 'st-now-op' }));
	// The letter tells the kind apart without relying on color.
	if (w >= 22 && (doneW >= w || doneW === 0)) out.push(text(x + w / 2, y + 13, `${KIND_LETTER[o.kind]}${o.chunk}`, `st-bar-text${doneW ? ` on-${o.kind}` : ''}`));
	return out;
}

export function render(root: SVGSVGElement, p: Params, st: State, width: number) {
	const narrow = width < 440;
	const lw = narrow ? 40 : 86;
	const x0 = PAD + lw;
	const cw = width - x0 - PAD;
	const ops = schedule(p);
	const serial = schedule(serialParams(p));
	const end = Math.max(makespan(ops), makespan(serial));
	const sx = (s: number) => x0 + (s / end) * cw;
	const t = st.time;
	const out: SVGElement[] = [
		svg('defs', {}, [
			svg('pattern', { id: 'st-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, [
				svg('rect', { width: 6, height: 6, class: 'st-hatch-bg' }),
				svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, class: 'st-hatch-line' }),
			]),
		]),
	];

	let y = PAD;
	const section = (title: string) => {
		out.push(text(PAD, y + 12, title, 'sim-label'));
		y += HEAD;
	};
	const row = (label: string, list: Op[], tt = t) => {
		out.push(svg('line', { x1: x0, y1: y + ROW - 4, x2: x0 + cw, y2: y + ROW - 4, class: 'st-lane' }));
		out.push(text(PAD, y + 14, label));
		for (const o of list) out.push(...bar(o, tt, y, sx));
		if (tt === t && t >= 0) out.push(svg('line', { x1: sx(t), y1: y - 3, x2: sx(t), y2: y + ROW - 1, class: 'st-cursor' }));
		y += ROW;
	};

	section(narrow ? 'By stream' : `By stream: chunk c goes to stream c % ${p.streams}`);
	for (let s = 0; s < p.streams; s++) row(narrow ? `s${s}` : `stream ${s}`, ops.filter((o) => o.stream === s));
	y += 6;
	section(narrow ? 'By engine' : 'By engine: what the hardware runs');
	const engines: Engine[] = p.engines === 2 ? ['h2d', 'compute', 'd2h'] : ['copy', 'compute'];
	for (const e of engines) row(ENGINE_LABEL[e][narrow ? 1 : 0], ops.filter((o) => o.engine === e));
	y += 6;
	section(narrow ? 'Serial' : 'Serial: one chunk, one stream, pinned');
	// The serial run is a reference, drawn complete.
	row(narrow ? 'ser' : 'serial', serial, Infinity);
	y += 4;
	out.push(svg('line', { x1: x0, y1: y, x2: x0 + cw, y2: y, class: 'st-axis' }));
	out.push(text(x0, y + 14, '0'));
	const endLabel = fmtTime(end);
	out.push(svg('text', { x: x0 + cw, y: y + 14, class: 'sim-small st-end' }, [endLabel]));
	y += 22;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...out);
}
