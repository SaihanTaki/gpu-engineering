import { clear, svg } from '@sim-core/svg';
import { gpuById } from '@sim-core/gpu-specs';
import {
	allocated,
	ARRAYS,
	arrayBytes,
	final,
	fmtBytes,
	fmtUs,
	kernelAt,
	LAST_STEP,
	PIPELINE_IDS,
	PIPELINE_NAMES,
	PIPELINES,
	totalsAt,
	type Config,
	type PipelineId,
} from './model';

const PAD = 12;
const GAP = 6;
const BOX_H = 30;
const BAR_H = 16;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: 2, class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

function caption(p: PipelineId, step: number) {
	if (step === 0) return 'not started';
	const i = kernelAt(p, step);
	if (i === null) return 'done';
	const k = PIPELINES[p][i];
	const of = p === 'fused' ? '' : `kernel ${i + 1} of 3: `;
	return `${of}${k.name}, ${k.math}`;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	const boxW = (inner - (ARRAYS.length - 1) * GAP) / ARRAYS.length;
	// Both pipelines share one scale, set by the slower one, so their bars compare directly.
	const maxBytes = final(c, 'unfused').bytes;
	const maxUs = Math.max(final(c, 'unfused').timeUs, final(c, 'fused').timeUs);
	const n = 2 ** c.log2n;
	const bw = gpuById(c.gpu).memoryGbPerS;
	let y = PAD;

	for (const p of PIPELINE_IDS) {
		nodes.push(text(PAD, y + 12, PIPELINE_NAMES[p]));
		nodes.push(text(PAD, y + 28, caption(p, step), 'sim-small'));
		y += 38;

		const i = kernelAt(p, step);
		const k = i === null ? null : PIPELINES[p][i];
		const has = allocated(p);
		ARRAYS.forEach((a, j) => {
			const x = PAD + j * (boxW + GAP);
			const cx = x + boxW / 2;
			if (!has.includes(a)) {
				nodes.push(rect(x, y, boxW, BOX_H, 'fu-box is-absent'));
				nodes.push(text(cx, y + 19, a, 'sim-small', 'middle'));
				nodes.push(text(cx, y + BOX_H + 13, 'none', 'sim-small', 'middle'));
				return;
			}
			const role = k?.reads.includes(a) ? 'read' : k?.writes.includes(a) ? 'write' : '';
			nodes.push(rect(x, y, boxW, BOX_H, `fu-box${role ? ` is-${role}` : ''}`));
			nodes.push(text(cx, y + 20, a, `sim-label${role === 'read' ? ' fu-on-read' : ''}`, 'middle'));
			if (role) nodes.push(text(cx, y + BOX_H + 13, role, 'sim-small fu-role', 'middle'));
		});
		y += BOX_H + 26;

		const t = totalsAt(c, p, step);
		nodes.push(text(PAD, y + 10, `DRAM traffic ${fmtBytes(t.bytes)}`, 'sim-small fu-bar-label'));
		y += 16;
		nodes.push(rect(PAD, y, inner, BAR_H, 'fu-track'));
		const rw = (inner * t.readBytes) / maxBytes;
		nodes.push(rect(PAD, y, rw, BAR_H, 'fu-read-bar'));
		nodes.push(rect(PAD + rw, y, (inner * t.writeBytes) / maxBytes, BAR_H, 'fu-write-bar'));
		y += BAR_H + 6;
		nodes.push(text(PAD, y + 10, `read ${fmtBytes(t.readBytes)}, written ${fmtBytes(t.writeBytes)}`, 'sim-small'));
		y += 22;

		nodes.push(text(PAD, y + 10, `Time (model) ${step === 0 ? '0' : fmtUs(t.timeUs)}`, 'sim-small fu-bar-label'));
		y += 16;
		nodes.push(rect(PAD, y, inner, BAR_H, 'fu-track'));
		// One launch segment, then one memory segment, per kernel run so far.
		let x = PAD;
		for (const kk of PIPELINES[p].slice(0, t.launches)) {
			const lw = (inner * c.launchUs) / maxUs;
			nodes.push(rect(x, y, lw, BAR_H, 'fu-launch'));
			x += lw;
			const memUs = ((kk.reads.length + kk.writes.length) * arrayBytes(n)) / (bw * 1e3);
			const mw = (inner * memUs) / maxUs;
			nodes.push(rect(x, y, mw, BAR_H, 'fu-mem'));
			x += mw;
		}
		y += BAR_H + 6;
		nodes.push(text(PAD, y + 10, `launch ${fmtUs(t.launchUs)}, memory ${fmtUs(t.memoryUs)}`, 'sim-small'));
		y += 20;

		if (p === 'unfused') {
			nodes.push(svg('line', { x1: PAD, x2: PAD + inner, y1: y + 4, y2: y + 4, class: 'fu-rule' }));
			y += 16;
		}
	}

	if (step === LAST_STEP) {
		const ratio = final(c, 'unfused').timeUs / final(c, 'fused').timeUs;
		nodes.push(text(PAD, y + 14, `Fused is ${ratio.toFixed(2)}x faster (model)`));
		y += 24;
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}
