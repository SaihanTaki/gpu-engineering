import { clear, svg } from '@sim-core/svg';
import { fmtBytes, naiveTraffic, flashTraffic, tileState, type Config, type Run } from './model';

const PAD = 12;
const BAR_H = 14;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-small', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);
const line = (x1: number, y1: number, x2: number, y2: number, cls: string) => svg('line', { x1, y1, x2, y2, class: cls });

export function render(root: SVGSVGElement, c: Config, run: Run, step: number, width: number) {
	const nodes: SVGElement[] = [];
	const cur = run.frames[step].step;
	const ci = cur.kind === 'start' ? -1 : cur.i;
	const cj = cur.kind === 'tile' ? cur.j : -1;
	const gap = 14;
	const cell = Math.max(6, Math.min(24, Math.floor((width - 2 * PAD - 2 * gap) / (c.n + 2 * c.d))));
	const qx = PAD;
	const sx = qx + c.d * cell + gap;
	const ox = sx + c.n * cell + gap;
	const used = ox + c.d * cell + PAD;
	// Center the drawing when the cells hit their maximum size.
	const shift = Math.max(0, (width - used) / 2);
	const X = (x: number) => x + shift;

	let y = PAD + 10;
	nodes.push(text(X(sx), y, 'Kᵀ (d x n)'));
	y += 6;
	const ky = y;
	const vy = ky + c.d * cell + 24 + c.n * cell + 8;
	const sy = ky + c.d * cell + 24;
	nodes.push(text(X(qx), sy - 6, 'Q'), text(X(sx), sy - 6, 'S = QKᵀ / √d'), text(X(ox), sy - 6, 'O'));

	const kvCls = (col: number) => (Math.floor(col / c.bc) === cj ? 'at-cell is-sram' : 'at-cell');
	for (let col = 0; col < c.n; col++) {
		for (let e = 0; e < c.d; e++) {
			nodes.push(rect(X(sx + col * cell), ky + e * cell, cell, cell, kvCls(col)));
			nodes.push(rect(X(sx + col * cell), vy + e * cell, cell, cell, kvCls(col)));
		}
	}
	for (let r = 0; r < c.n; r++) {
		const inBlock = Math.floor(r / c.br) === ci;
		const ws = run.writeStep[Math.floor(r / c.br)];
		const oCls = ws < step ? 'at-cell is-done' : ws === step ? 'at-cell is-sram' : inBlock ? 'at-cell is-acc' : 'at-cell';
		for (let e = 0; e < c.d; e++) {
			nodes.push(rect(X(qx + e * cell), sy + r * cell, cell, cell, inBlock && cur.kind === 'tile' ? 'at-cell is-sram' : 'at-cell'));
			nodes.push(rect(X(ox + e * cell), sy + r * cell, cell, cell, oCls));
		}
		for (let col = 0; col < c.n; col++) {
			const st = tileState(run, Math.floor(r / c.br), Math.floor(col / c.bc), step);
			const masked = c.causal && col > r;
			const cls = st === 'skipped' || (masked && st !== 'pending') ? 'is-masked' : st === 'current' ? 'is-sram' : st === 'done' ? 'is-done' : '';
			nodes.push(rect(X(sx + col * cell), sy + r * cell, cell, cell, `at-cell ${cls}`));
		}
	}

	// Block boundaries, then a cross on every skipped tile and an outline on the current one.
	const tr = Math.ceil(c.n / c.br);
	const tc = Math.ceil(c.n / c.bc);
	for (let i = 1; i < tr; i++) {
		const yy = sy + i * c.br * cell;
		nodes.push(line(X(qx), yy, X(qx + c.d * cell), yy, 'at-edge'), line(X(sx), yy, X(sx + c.n * cell), yy, 'at-edge'));
		nodes.push(line(X(ox), yy, X(ox + c.d * cell), yy, 'at-edge'));
	}
	for (let j = 1; j < tc; j++) {
		const xx = X(sx + j * c.bc * cell);
		nodes.push(line(xx, ky, xx, ky + c.d * cell, 'at-edge'), line(xx, sy, xx, sy + c.n * cell, 'at-edge'));
		nodes.push(line(xx, vy, xx, vy + c.d * cell, 'at-edge'));
	}
	for (let i = 0; i < tr; i++) {
		for (let j = 0; j < tc; j++) {
			const x0 = X(sx + j * c.bc * cell);
			const y0 = sy + i * c.br * cell;
			const w = (Math.min(c.n, (j + 1) * c.bc) - j * c.bc) * cell;
			const h = (Math.min(c.n, (i + 1) * c.br) - i * c.br) * cell;
			if (run.tileStep[i][j] < 0) nodes.push(line(x0, y0, x0 + w, y0 + h, 'at-skip'), line(x0 + w, y0, x0, y0 + h, 'at-skip'));
		}
	}
	if (cur.kind === 'tile') {
		const x0 = X(sx + cur.j * c.bc * cell);
		const y0 = sy + cur.i * c.br * cell;
		const w = (Math.min(c.n, (cur.j + 1) * c.bc) - cur.j * c.bc) * cell;
		const h = (Math.min(c.n, (cur.i + 1) * c.br) - cur.i * c.br) * cell;
		for (const [x, yy, ww, hh] of [
			[x0, y0, w, h],
			[X(qx), y0, c.d * cell, h],
			[x0, ky, w, c.d * cell],
			[x0, vy, w, c.d * cell],
		]) {
			nodes.push(rect(x, yy, ww, hh, 'at-current'), rect(x + 2.25, yy + 2.25, ww - 4.5, hh - 4.5, 'at-current-ring'));
		}
	}
	for (const [x, yy, w, h] of [
		[qx, sy, c.d, c.n],
		[sx, ky, c.n, c.d],
		[sx, sy, c.n, c.n],
		[sx, vy, c.n, c.d],
		[ox, sy, c.d, c.n],
	]) {
		nodes.push(rect(X(x), yy, w * cell, h * cell, 'at-frame'));
	}
	y = vy + c.d * cell + 16;
	nodes.push(text(X(sx), y, 'Vᵀ (d x n)'));
	y += 20;

	nodes.push(text(PAD, y, 'HBM traffic', 'sim-label'));
	y += 20;
	// HBM traffic: flash so far against the whole naive pass, on one scale.
	const naive = naiveTraffic(c);
	const naiveTotal = naive.read + naive.written;
	const f = run.frames[step];
	const flashTotal = flashTraffic(c);
	const scale = Math.max(naiveTotal, flashTotal.read + flashTotal.written);
	const inner = width - 2 * PAD;
	const bar = (label: string, read: number, written: number) => {
		nodes.push(text(PAD, y, `${label}: ${fmtBytes(read + written)}`, 'sim-small at-bar-label'));
		y += 6;
		nodes.push(rect(PAD, y, inner, BAR_H, 'at-track'));
		const rw = (inner * read) / scale;
		nodes.push(rect(PAD, y, rw, BAR_H, 'at-read'), rect(PAD + rw, y, (inner * written) / scale, BAR_H, 'at-write'));
		y += BAR_H + 14;
		nodes.push(text(PAD, y, `read ${fmtBytes(read)}, written ${fmtBytes(written)}`));
		y += 22;
	};
	bar('Naive, whole pass', naive.read, naive.written);
	bar('Flash, so far', f.read, f.written);
	y -= 8;

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
