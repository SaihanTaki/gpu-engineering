import { clear, svg } from '@sim-core/svg';
import { AXIS_MAX, AXIS_MIN, FORMATS, maxFinite, minNormal, minSubnormal } from './model';

const ROW = 28;
const LABEL_W = 70;
const M = { top: 8, right: 22, bottom: 34 };

const text = (x: number, y: number, s: string, cls = 'sim-small', anchor?: string) => svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

/**
 * One bar per format on a log2 axis: the subnormal range, then the normal
 * range up to the largest finite value, and a line at |x|.
 */
export function renderRange(root: SVGSVGElement, x: number, editing: string, width: number) {
	const height = M.top + FORMATS.length * ROW + M.bottom;
	const left = LABEL_W;
	const pw = width - left - M.right;
	const X = (e: number) => left + ((Math.min(AXIS_MAX, Math.max(AXIS_MIN, e)) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * pw;
	const bottom = M.top + FORMATS.length * ROW;
	const nodes: SVGElement[] = [];

	const every = pw < 360 ? 64 : 32;
	for (let e = Math.ceil(AXIS_MIN / every) * every; e <= AXIS_MAX; e += every) {
		nodes.push(svg('line', { x1: X(e), y1: M.top, x2: X(e), y2: bottom, class: 'bl-grid' }));
		nodes.push(text(X(e), bottom + 14, e === 0 ? '1' : `2^${e}`, 'sim-small', 'middle'));
	}
	nodes.push(text(left + pw / 2, height - 4, '|value|, log scale', 'sim-small bl-axis', 'middle'));

	FORMATS.forEach((f, i) => {
		const y = M.top + i * ROW + 6;
		const h = ROW - 12;
		const sub0 = X(Math.log2(minSubnormal(f)));
		const norm0 = X(Math.log2(minNormal(f)));
		const end = X(Math.log2(maxFinite(f)));
		const on = f.id === editing;
		nodes.push(text(4, y + h - 3, f.name, `sim-small bl-row-label${on ? ' is-editing' : ''}`));
		nodes.push(svg('rect', { x: sub0, y, width: Math.max(1, norm0 - sub0), height: h, class: 'bl-sub' }));
		nodes.push(svg('rect', { x: norm0, y, width: Math.max(1, end - norm0), height: h, class: `bl-norm${on ? ' is-editing' : ''}` }));
	});

	const a = Math.abs(x);
	if (Number.isFinite(a) && a > 0) {
		const lx = X(Math.log2(a));
		nodes.push(svg('line', { x1: lx, y1: M.top - 4, x2: lx, y2: bottom + 2, class: 'bl-marker' }));
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...nodes);
}
