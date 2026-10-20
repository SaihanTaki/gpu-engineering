import { clear, svg } from '@sim-core/svg';
import { gpuById } from '@sim-core/gpu-specs';
import {
	accesses,
	accessIndexAt,
	BANKS,
	estimate,
	fmt,
	globalCost,
	isSync,
	KERNEL_IDS,
	KERNEL_NAMES,
	LAST_STEP,
	sharedCost,
	totalsAt,
	WORDS_PER_SECTOR,
	type Access,
	type Config,
	type KernelId,
} from './model';

const PAD = 12;
const GAP = 3;
const SECTOR_H = 14;
const UNIT = 3;
const BAR_H = 16;

const rect = (x: number, y: number, w: number, h: number, cls: string) =>
	svg('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), class: cls });
const text = (x: number, y: number, s: string, cls = 'sim-label', anchor?: string) =>
	svg('text', { x, y, class: cls, 'text-anchor': anchor }, [s]);

const lanesOn = (a: Access) => a.index.filter((i) => i !== null).length;
const conflict = (d: number) => (d <= 1 ? 'no conflict' : `${d}-way conflict`);

function caption(a: Access): string {
	if (lanesOn(a) === 0) return 'all lanes out of bounds';
	switch (a.kind) {
		case 'global-load':
			return `read in: ${globalCost(a).sectors.length} sectors`;
		case 'global-store':
			return `write out: ${globalCost(a).sectors.length} sectors`;
		case 'shared-store':
			return `tile row: ${conflict(sharedCost(a).degree)}`;
		case 'shared-load':
			return `tile column: ${conflict(sharedCost(a).degree)}`;
	}
}

/** Sectors as boxes of 8 word slots; used words filled. 32, 16 or 8 boxes per row. */
const sectorsPerRow = (inner: number) => [32, 16, 8].find((n) => (inner - (n - 1) * GAP) / n >= 16) ?? 8;

function drawSectors(nodes: SVGElement[], a: Access, top: number, inner: number) {
	const sectors = globalCost(a).sectors;
	const perRow = sectorsPerRow(inner);
	const w = Math.min(48, Math.floor((inner - (perRow - 1) * GAP) / perRow));
	const slot = w / WORDS_PER_SECTOR;
	sectors.forEach((s, i) => {
		const x = PAD + (i % perRow) * (w + GAP);
		const y = top + Math.floor(i / perRow) * (SECTOR_H + GAP);
		s.words.forEach((used, j) => {
			if (used) nodes.push(rect(x + j * slot, y, slot, SECTOR_H, 'tr-word'));
		});
		nodes.push(rect(x, y, w, SECTOR_H, 'tr-sector'));
	});
}

/** One column per bank; its height is the distinct words that bank must serve. */
function drawBanks(nodes: SVGElement[], a: Access, base: number, inner: number) {
	const { perBank } = sharedCost(a);
	const colW = (inner - (BANKS - 1)) / BANKS;
	perBank.forEach((count, b) => {
		const x = PAD + b * (colW + 1);
		nodes.push(rect(x, base - UNIT, colW, UNIT, 'tr-bank'));
		if (count) nodes.push(rect(x, base - count * UNIT, colW, count * UNIT, count > 1 ? 'tr-bank-fill is-conflict' : 'tr-bank-fill'));
	});
	nodes.push(text(PAD, base + 13, 'bank 0', 'sim-small'), text(PAD + inner, base + 13, '31', 'sim-small', 'end'));
	const worst = perBank.indexOf(Math.max(...perBank));
	if (perBank[worst] > 1) {
		const x = Math.min(PAD + worst * (colW + 1) + colW + 4, PAD + inner - 70);
		nodes.push(text(x, base - perBank[worst] * UNIT + 10, `${perBank[worst]} words`, 'sim-small tr-costly'));
	}
}

/** Height a request's drawing needs, so a row is only as tall as the step requires. */
function visHeight(a: Access | null, inner: number) {
	if (!a || !lanesOn(a)) return 0;
	if (a.kind === 'global-load' || a.kind === 'global-store') {
		const rows = Math.ceil(globalCost(a).sectors.length / sectorsPerRow(inner));
		return rows * (SECTOR_H + GAP) - GAP;
	}
	return Math.max(...sharedCost(a).perBank, 1) * UNIT + 16;
}

export function render(root: SVGSVGElement, c: Config, step: number, width: number) {
	const inner = width - 2 * PAD;
	const nodes: SVGElement[] = [];
	let y = PAD;

	const current = KERNEL_IDS.map((k) => {
		const i = accessIndexAt(k, step);
		return i === null ? null : accesses(c, k)[i];
	});
	// Rows that draw something share one height, so the kernels stay comparable.
	const visH = Math.max(...current.map((a) => visHeight(a, inner)));

	KERNEL_IDS.forEach((k, ki) => {
		const a = current[ki];
		nodes.push(text(PAD, y + 12, KERNEL_NAMES[k]));
		const what = step === 0 ? 'not started' : isSync(k, step) ? '__syncthreads()' : a ? caption(a) : 'done';
		nodes.push(text(PAD, y + 28, what, a && isCostly(a) ? 'sim-small tr-costly' : 'sim-small'));
		const top = y + 38;
		if (a && lanesOn(a)) {
			if (a.kind === 'global-load' || a.kind === 'global-store') drawSectors(nodes, a, top, inner);
			else drawBanks(nodes, a, top + visH - 16, inner);
		}
		// A kernel with nothing to draw this step keeps only its caption.
		y = top + (a && lanesOn(a) ? visH : -8) + 10;
		nodes.push(svg('line', { x1: PAD, x2: PAD + inner, y1: y - 5, y2: y - 5, class: 'tr-rule' }));
	});

	// Bandwidth bars, revealed at the last step so the reader predicts first.
	const g = gpuById(c.gpu);
	y += 6;
	nodes.push(text(PAD, y + 12, 'Estimated bandwidth (model)'));
	nodes.push(text(PAD, y + 28, `not a measurement; peak ${fmt(g.memoryGbPerS)} GB/s`, 'sim-small'));
	y += 40;
	for (const k of KERNEL_IDS) {
		const e = estimate(g, totalsAt(c, k, step));
		const shown = step === LAST_STEP;
		const label = !shown ? `${KERNEL_NAMES[k]}: ?` : e.gbPerS === null ? `${KERNEL_NAMES[k]}: no data moved` : `${KERNEL_NAMES[k]}: ${fmt(e.gbPerS)} GB/s`;
		nodes.push(text(PAD, y + 12, label, 'sim-small tr-bar-label'));
		y += 18;
		nodes.push(rect(PAD, y, inner, BAR_H, 'tr-bar-track'));
		if (shown && e.gbPerS !== null) nodes.push(rect(PAD, y, (inner * e.gbPerS) / g.memoryGbPerS, BAR_H, 'tr-bar'));
		y += BAR_H + 4;
		const why = !shown ? 'step to the end to see it' : e.limit === 'shared' ? 'limited by shared memory passes' : e.limit === 'global' ? 'limited by sector traffic' : '';
		nodes.push(text(PAD, y + 10, why, 'sim-small'));
		y += 22;
	}

	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${y}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(y));
	root.append(...nodes);
}

/** More than the 4 sectors or 1 pass a fully coalesced, conflict-free request needs. */
function isCostly(a: Access) {
	if (a.kind === 'global-load' || a.kind === 'global-store') return globalCost(a).sectors.length > 5;
	return sharedCost(a).degree > 1;
}
