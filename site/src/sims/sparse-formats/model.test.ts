import { describe as suite, expect, it } from 'vitest';
import {
	csr,
	describe,
	ell,
	KERNEL_IDS,
	lastStep,
	lengths,
	lineAt,
	N,
	nnz,
	pattern,
	phaseAt,
	PRESET_IDS,
	readsAt,
	sectorsAt,
	totals,
	value,
} from './model';

suite('formats', () => {
	it('row lengths of the presets', () => {
		expect(lengths('banded')).toEqual([2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]);
		expect(lengths('random')).toEqual([2, 3, 1, 4, 2, 3, 5, 2, 3, 1, 4, 2]);
		// ceil(12 / (r + 1)).
		expect(lengths('power-law')).toEqual([12, 6, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1]);
		expect(PRESET_IDS.map(nnz)).toEqual([34, 32, 41]);
	});

	it('columns are distinct, sorted and in range', () => {
		for (const p of PRESET_IDS)
			for (const cols of pattern(p)) {
				expect(new Set(cols).size).toBe(cols.length);
				expect([...cols].sort((a, b) => a - b)).toEqual(cols);
				expect(cols.every((c) => c >= 0 && c < N)).toBe(true);
			}
	});

	it('CSR is COO with the row array compressed', () => {
		for (const p of PRESET_IDS) {
			const a = csr(p);
			expect(a.rowPtr).toHaveLength(N + 1);
			expect(a.rowPtr[N]).toBe(nnz(p));
			for (let r = 0; r < N; r++)
				for (let i = a.rowPtr[r]; i < a.rowPtr[r + 1]; i++) {
					expect(a.row[i]).toBe(r);
					expect(a.val[i]).toBe(value(r, a.col[i]));
				}
		}
		expect(csr('banded').rowPtr).toEqual([0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34]);
	});

	it('ELL pads every row to the longest, column-major', () => {
		const e = ell('power-law');
		expect(e.width).toBe(12);
		expect(e.pad.filter(Boolean)).toHaveLength(144 - 41);
		const b = ell('banded');
		expect(b.width).toBe(3);
		// Slot 0 of rows 0 and 1 are the first two entries; row 0 has columns 0 and 1, row 1 columns 0, 1, 2.
		expect(b.col.slice(0, 2)).toEqual([0, 0]);
		expect(b.col[1 * N + 0]).toBe(1);
		// Row 0 has no third entry, row 11 neither.
		expect([b.pad[2 * N + 0], b.pad[2 * N + 11], b.pad[2 * N + 5]]).toEqual([true, true, false]);
		expect(b.val[2 * N]).toBe(0);
	});
});

suite('steps and reads', () => {
	it('scalar and ELL loop as long as the longest row, vector does 1 pass and 5 shuffles', () => {
		expect(lastStep('scalar', 'power-law')).toBe(13);
		expect(lastStep('ell', 'banded')).toBe(4);
		expect(lastStep('vector', 'power-law')).toBe(7);
		expect([2, 3, 4, 5, 6, 7].map((s) => phaseAt('vector', 'banded', s))).toEqual([
			{ kind: 'shuffle', offset: 16 },
			{ kind: 'shuffle', offset: 8 },
			{ kind: 'shuffle', offset: 4 },
			{ kind: 'shuffle', offset: 2 },
			{ kind: 'shuffle', offset: 1 },
			{ kind: 'write' },
		]);
	});

	it('every nonzero is read exactly once by each kernel', () => {
		for (const p of PRESET_IDS)
			for (const k of KERNEL_IDS) {
				const seen: number[] = [];
				for (let s = 0; s <= lastStep(k, p); s++) for (const r of readsAt(k, p, s)) if (r.csrIndex !== null) seen.push(r.csrIndex);
				expect(seen.sort((a, b) => a - b)).toEqual(Array.from({ length: nnz(p) }, (_, i) => i));
			}
	});

	it('scalar on banded: iteration 1 reads row_ptr[r], 5 sectors', () => {
		const reads = readsAt('scalar', 'banded', 1);
		expect(reads.map((r) => r.address)).toEqual([0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]);
		// Sectors 0, 1, 2, 3, 4 of 8 entries each.
		expect(sectorsAt('scalar', 'banded', 1)).toBe(5);
		// Only rows 1 to 10 have a third entry: 4, 7, ..., 31 fall in sectors 0 to 3.
		expect(readsAt('scalar', 'banded', 3)).toHaveLength(10);
		expect(sectorsAt('scalar', 'banded', 3)).toBe(4);
	});

	it('ELL reads 12 contiguous entries per step: 2 sectors', () => {
		for (let s = 1; s <= 3; s++) {
			expect(readsAt('ell', 'banded', s).map((r) => r.address)).toEqual(Array.from({ length: N }, (_, r) => (s - 1) * N + r));
			expect(sectorsAt('ell', 'banded', s)).toBe(2);
		}
	});

	it('vector: one request per warp; rows 5 and 8 of banded cross a sector boundary', () => {
		// Row 5 is entries 14..16, row 8 is 23..25; every other row fits one sector.
		expect(sectorsAt('vector', 'banded', 1)).toBe(14);
		expect(readsAt('vector', 'power-law', 1).filter((r) => r.row === 0).map((r) => r.lane)).toEqual([...Array(12).keys()]);
		expect(readsAt('vector', 'banded', 2)).toEqual([]);
	});
});

suite('totals', () => {
	it('banded: little waste anywhere', () => {
		expect(totals('scalar', 'banded')).toEqual({ steps: 3, useful: 34, slots: 36, padding: 0, idle: 2, sectors: 14 });
		expect(totals('ell', 'banded')).toEqual({ steps: 3, useful: 34, slots: 36, padding: 2, idle: 0, sectors: 6 });
		expect(totals('vector', 'banded')).toEqual({ steps: 6, useful: 34, slots: 384, padding: 0, idle: 350, sectors: 14 });
	});

	it('power-law: scalar waits for row 0, ELL pads 103 slots', () => {
		const s = totals('scalar', 'power-law');
		expect([s.steps, s.slots, s.idle]).toEqual([12, 144, 103]);
		const e = totals('ell', 'power-law');
		expect([e.steps, e.slots, e.padding]).toEqual([12, 144, 103]);
		// 12 steps of 12 contiguous entries: 2 sectors each.
		expect(e.sectors).toBe(24);
		expect(totals('vector', 'power-law').steps).toBe(6);
	});
});

suite('text and code', () => {
	it('every step of every kernel has a status line', () => {
		for (const p of PRESET_IDS)
			for (const k of KERNEL_IDS)
				for (let s = 0; s <= lastStep(k, p); s++) expect(describe(k, p, s).length).toBeGreaterThan(40);
		expect(describe('ell', 'power-law', 13)).toContain('103 padding slots');
		expect(describe('scalar', 'power-law', 12)).toContain('11 have finished');
	});

	it('maps steps to code lines', () => {
		expect([0, 1, 4].map((s) => lineAt('scalar', 'banded', s))).toEqual([null, 5, 6]);
		expect([1, 2, 6, 7].map((s) => lineAt('vector', 'banded', s))).toEqual([6, 8, 8, 9]);
		expect([1, 4].map((s) => lineAt('ell', 'banded', s))).toEqual([6, 8]);
	});
});
