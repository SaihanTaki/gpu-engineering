import { describe as suite, expect, it } from 'vitest';
import { analyze, codeFor, describe, ELEM_BYTES, LOAD_LINE, PATTERN_IDS, stateAt, type Config } from './model';

const cfg = (o: Partial<Config>): Config => ({ pattern: 'strided', elemBytes: 4, stride: 1, offset: 0, fields: 3, ...o });
const count = (o: Partial<Config>) => {
	const a = analyze(cfg(o));
	return { sectors: a.sectors.length, lines: a.lines.length, eff: a.efficiency };
};

suite('coalescing', () => {
	// The cases the Best Practices Guide and the Programming Guide work through, for 4-byte words.
	it('consecutive floats: four sectors, one line, 100%', () => {
		expect(count({})).toEqual({ sectors: 4, lines: 1, eff: 1 });
	});

	it('offset by one float: five sectors over two lines', () => {
		expect(count({ offset: 1 })).toEqual({ sectors: 5, lines: 2, eff: 0.8 });
	});

	it('offsets that are multiples of 8 words stay at four sectors', () => {
		for (const offset of [8, 16, 24, 32]) expect(count({ offset }).sectors).toBe(4);
		for (const offset of [1, 2, 7, 9, 31]) expect(count({ offset }).sectors).toBe(5);
	});

	it('stride 2 halves the efficiency; stride 8 or more needs one sector per lane', () => {
		expect(count({ stride: 2 })).toEqual({ sectors: 8, lines: 2, eff: 0.5 });
		for (const stride of [8, 9, 16, 32]) expect(count({ stride })).toMatchObject({ sectors: 32, eff: 0.125 });
		expect(count({ stride: 32 }).lines).toBe(32);
		expect(count({ stride: 8 }).lines).toBe(8);
	});

	it('element size sets the bytes per warp: 1 byte to 16 bytes', () => {
		expect(ELEM_BYTES.map((elemBytes) => count({ elemBytes }).sectors)).toEqual([1, 2, 4, 8, 16]);
		expect(count({ elemBytes: 16 }).lines).toBe(4);
		for (const elemBytes of ELEM_BYTES) expect(count({ elemBytes }).eff).toBe(1);
	});

	it('AoS of three floats touches 12 sectors; SoA touches 4', () => {
		const aos = analyze(cfg({ pattern: 'aos', fields: 3 }));
		expect(aos.sectors).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
		expect(aos.usedBytes).toBe(128);
		expect(aos.efficiency).toBeCloseTo(1 / 3, 10);
		expect(count({ pattern: 'soa', fields: 3 })).toEqual({ sectors: 4, lines: 1, eff: 1 });
		// SoA ignores the stride and offset sliders.
		expect(count({ pattern: 'soa', stride: 7, offset: 3 }).sectors).toBe(4);
	});

	it('no element ever spans two sectors', () => {
		for (const elemBytes of ELEM_BYTES)
			for (let stride = 1; stride <= 32; stride++)
				for (let offset = 0; offset <= 32; offset++)
					for (const a of analyze(cfg({ elemBytes, stride, offset })).accesses)
						expect(Math.floor((a.addr + elemBytes - 1) / 32)).toBe(a.sector);
	});

	it('stepping reveals sectors one lane at a time', () => {
		const c = cfg({ offset: 1 });
		expect(stateAt(c, 0).sofar.sectors).toEqual([]);
		expect(stateAt(c, 7).sofar.sectors).toEqual([0]);
		expect(stateAt(c, 8).reused).toBe(false);
		expect(stateAt(c, 8).sofar.sectors).toEqual([0, 1]);
		expect(stateAt(c, 9).reused).toBe(true);
		expect(stateAt(c, 32).sofar.sectors.length).toBe(5);
	});

	it('every step has a status line', () => {
		for (const pattern of PATTERN_IDS) {
			const c = cfg({ pattern });
			for (let s = 0; s <= 32; s++) expect(describe(c, s).length).toBeGreaterThan(40);
			expect(codeFor(pattern).split('\n')[LOAD_LINE - 1]).toMatch(/T v = /);
		}
		expect(describe(cfg({}), 0)).toContain('Predict');
		expect(describe(cfg({}), 32)).toContain('Done: 4 sectors in 1 line: 128 bytes fetched for 128 bytes used, 100% efficiency.');
		expect(describe(cfg({ offset: 1 }), 8)).toBe('Lane 7 reads bytes 32 to 35, in sector 1 of line 0. That sector is new: the warp now needs 2 sectors.');
	});
});
