import { clear, svg } from '@sim-core/svg';
import { LINE, SECTOR, WARP, type Config, type State } from './model';

const PAD = 12;
const GAP = 4;
const LABEL_W = 56;
const SECTOR_H = 30;

/** 32, 16 or 8 lanes per row, whichever is widest with cells of at least 26px. */
const lanesPerRow = (inner: number) => [32, 16, 8].find((n) => (inner - (n - 1) * GAP) / n >= 26) ?? 8;

export function render(root: SVGSVGElement, c: Config, st: State, width: number) {
	const inner = width - 2 * PAD;
	const out: SVGElement[] = [];
	const cur = st.current;
	const placed = st.sofar.accesses;
	const fetched = new Set(st.sofar.sectors);

	out.push(
		svg('text', { x: PAD, y: 16, class: 'sim-label' }, [
			cur ? `lane ${cur.lane} -> sector ${cur.sector}` : 'one warp, 32 lanes, one load each',
		]),
	);

	const perRow = lanesPerRow(inner);
	const cell = Math.min(40, Math.floor((inner - (perRow - 1) * GAP) / perRow));
	const rowH = cell + 18;
	const top = 26;
	for (let lane = 0; lane < WARP; lane++) {
		const x = PAD + (lane % perRow) * (cell + GAP);
		const y = top + Math.floor(lane / perRow) * rowH;
		const a = lane < placed.length ? placed[lane] : null;
		const isCur = cur?.lane === lane;
		out.push(
			svg('text', { x: x + cell / 2, y: y + 10, class: 'sim-index' }, [String(lane)]),
			svg('rect', { x, y: y + 14, width: cell, height: cell, rx: 3, class: `co-lane${a ? ' is-placed' : ''}${isCur ? ' is-current' : ''}` }),
		);
		if (a) out.push(svg('text', { x: x + cell / 2, y: y + 14 + cell / 2 + 4, class: `co-lane-text${isCur ? ' is-current' : ''}` }, [String(a.sector)]));
	}
	let y = top + Math.ceil(WARP / perRow) * rowH + 18;

	out.push(svg('text', { x: PAD, y, class: 'sim-label' }, [`sectors fetched: ${st.sofar.sectors.length}, lines: ${st.sofar.lines.length}`]));
	y += 8;

	// Every line the whole warp will touch is drawn from step 0, so the drawing keeps its size while stepping.
	const lines = st.full.lines;
	const minLineW = LABEL_W + 4 * 44;
	const linesPerRow = Math.max(1, Math.min(4, Math.floor((inner + 16) / (minLineW + 16))));
	const lineW = (inner - (linesPerRow - 1) * 16) / linesPerRow;
	const sectorW = (lineW - LABEL_W - 3 * 2) / 4;
	lines.forEach((line, k) => {
		const lx = PAD + (k % linesPerRow) * (lineW + 16);
		const ly = y + Math.floor(k / linesPerRow) * (SECTOR_H + 8);
		out.push(svg('text', { x: lx, y: ly + 19, class: 'sim-small' }, [`+${line * LINE}`]));
		for (let s = 0; s < 4; s++) {
			const sector = line * 4 + s;
			const sx = lx + LABEL_W + s * (sectorW + 2);
			const cls = `co-sector${fetched.has(sector) ? ' is-fetched' : ''}${cur?.sector === sector ? ' is-current' : ''}`;
			out.push(svg('rect', { x: sx, y: ly, width: sectorW, height: SECTOR_H, class: cls }));
			for (const a of placed) {
				if (a.sector !== sector) continue;
				const ex = sx + ((a.addr - sector * SECTOR) / SECTOR) * sectorW;
				const ew = Math.max(1.5, (c.elemBytes / SECTOR) * sectorW - 0.5);
				out.push(svg('rect', { x: ex, y: ly + 3, width: ew, height: 13, class: `co-elem${a === cur ? ' is-current' : ''}` }));
			}
			out.push(svg('text', { x: sx + sectorW / 2, y: ly + SECTOR_H - 4, class: 'co-sector-text' }, [String(sector)]));
		}
	});
	y += Math.ceil(lines.length / linesPerRow) * (SECTOR_H + 8);

	const height = y + PAD;
	clear(root);
	root.setAttribute('viewBox', `0 0 ${width} ${height}`);
	root.setAttribute('width', String(width));
	root.setAttribute('height', String(height));
	root.append(...out);
}
