import { clear, svg } from '@sim-core/svg';
import { BANKS, listNumbers, serves, wordsPerBank, type Config, type State } from './model';

const PAD = 12;
const LABEL_W = 34;
const GAP = 2;
const CELL_H = 22;
const HEAD_H = 14;
const FOOT_H = 18;
const BAND_GAP = 12;

/** 32, 16 or 8 banks per band, whichever is widest with cells of at least 22px. */
const banksPerBand = (inner: number) => [32, 16, 8].find((n) => (inner - (n - 1) * GAP) / n >= 22) ?? 8;

export function render(root: SVGSVGElement, c: Config, st: State, width: number) {
	const inner = width - 2 * PAD - LABEL_W;
	const per = banksPerBand(inner);
	const cell = Math.min(40, Math.floor((inner - (per - 1) * GAP) / per));
	const x0 = PAD + LABEL_W;
	const xOf = (bank: number) => x0 + (bank % per) * (cell + GAP);
	const words = wordsPerBank(c);
	const servedSoFar = Array<number>(BANKS).fill(0);
	for (const s of st.done) servedSoFar[s.bank]++;
	const all = serves(c);

	const out: SVGElement[] = [
		svg('text', { x: PAD, y: 16, class: 'sim-label' }, [st.step ? `wavefront ${st.step}: one word per bank` : 'no wavefront yet']),
	];

	let y = 28;
	for (let b0 = 0; b0 < BANKS; b0 += per) {
		const banks = Array.from({ length: per }, (_, i) => b0 + i);
		// Rows for every wavefront this band will need, so the drawing does not grow while stepping.
		const depth = Math.max(...banks.map((b) => words[b]));
		out.push(svg('text', { x: PAD, y: y + 10, class: 'sim-small' }, ['bank']));
		for (const b of banks) out.push(svg('text', { x: xOf(b) + cell / 2, y: y + 10, class: 'sim-index' }, [String(b)]));
		const rowsTop = y + HEAD_H;
		const rowY = (wave: number) => rowsTop + wave * (CELL_H + GAP);

		for (let wave = 0; wave < Math.min(depth, st.step); wave++) {
			out.push(svg('text', { x: PAD, y: rowY(wave) + 15, class: 'sim-small' }, [`w${wave + 1}`]));
		}
		for (const s of all) {
			if (s.bank < b0 || s.bank >= b0 + per || s.wave >= st.step) continue;
			const x = xOf(s.bank);
			const cy = rowY(s.wave);
			const label = s.lanes.length > 1 ? `${s.lanes[0]}+` : String(s.lanes[0]);
			const tip = `bank ${s.bank}, word ${s.word}, ${s.lanes.length > 1 ? `lanes ${listNumbers(s.lanes)}` : `lane ${s.lanes[0]}`}`;
			out.push(
				svg('g', {}, [
					svg('title', {}, [tip]),
					svg('rect', { x, y: cy, width: cell, height: CELL_H, rx: 2, class: `bk-cell${s.wave > 0 ? ' is-replay' : ''}` }),
					svg('text', { x: x + cell / 2, y: cy + 15, class: `bk-cell-text${s.lanes.length > 1 ? ' is-shared' : ''}` }, [label]),
				]),
			);
		}
		if (st.step > 0 && st.step <= depth) {
			out.push(
				svg('rect', {
					x: x0 - 2,
					y: rowY(st.step - 1) - 2,
					width: per * (cell + GAP) - GAP + 4,
					height: CELL_H + 4,
					rx: 3,
					class: 'bk-now',
				}),
			);
		}

		const footY = rowY(depth) + 12;
		out.push(svg('text', { x: PAD, y: footY, class: 'sim-small' }, ['words']));
		for (const b of banks) {
			if (!servedSoFar[b]) continue;
			out.push(
				svg('text', { x: xOf(b) + cell / 2, y: footY, class: `bk-count${servedSoFar[b] > 1 ? ' is-conflict' : ''}` }, [
					servedSoFar[b] > 1 ? `${servedSoFar[b]}x` : '1',
				]),
			);
		}
		y = rowY(depth) + FOOT_H + BAND_GAP;
	}

	const height = y - BAND_GAP + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
