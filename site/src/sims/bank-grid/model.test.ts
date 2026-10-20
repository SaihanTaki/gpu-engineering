import { describe as suite, expect, it } from 'vitest';
import {
	accesses,
	BANKS,
	codeFor,
	degree,
	describe,
	lastStep,
	LAYOUT_IDS,
	listNumbers,
	READ_LINE,
	serves,
	SHARES,
	stateAt,
	STRIDES,
	type Config,
	type Layout,
} from './model';

const cfg = (stride: number, layout: Layout = 'plain', share = 1): Config => ({ stride, layout, share });
const all = STRIDES.flatMap((stride) => SHARES.flatMap((share) => LAYOUT_IDS.map((layout) => cfg(stride, layout, share))));

suite('bank grid', () => {
	it('the constants match the CUDA Programming Guide: 32 banks of 4 bytes', () => {
		expect(BANKS).toBe(32);
		expect(accesses(cfg(1)).map((a) => a.bank)).toEqual(Array.from({ length: 32 }, (_, i) => i));
	});

	it.each([
		[1, 1],
		[2, 2],
		[3, 1],
		[4, 4],
		[8, 8],
		[16, 16],
		[17, 1],
		[32, 32],
		[33, 1],
		[64, 32],
	])('plain tile, stride %i: %i-way', (stride, d) => {
		expect(degree(cfg(stride))).toBe(d);
	});

	it('a column read (stride 32) is 32-way plain and conflict-free with padding or swizzle', () => {
		expect(degree(cfg(32, 'plain'))).toBe(32);
		expect(degree(cfg(32, 'pad'))).toBe(1);
		expect(degree(cfg(32, 'swizzle'))).toBe(1);
		expect(accesses(cfg(32, 'pad')).map((a) => a.word)).toEqual(Array.from({ length: 32 }, (_, i) => i * 33));
	});

	it('padding does not fix every stride: stride 64 stays 2-way, and the diagonal stride 33 becomes 2-way', () => {
		expect(degree(cfg(64, 'pad'))).toBe(2);
		expect(degree(cfg(64, 'swizzle'))).toBe(2);
		expect(degree(cfg(33, 'pad'))).toBe(2);
		expect(degree(cfg(2, 'pad'))).toBe(1);
	});

	it('every lane reading one word is a broadcast: one wavefront, one serve with 32 lanes', () => {
		const s = serves(cfg(0));
		expect(degree(cfg(0))).toBe(1);
		expect(s).toHaveLength(1);
		expect(s[0].lanes).toHaveLength(32);
	});

	it('pairs of lanes sharing a word halve the column conflict', () => {
		expect(degree(cfg(32, 'plain', 2))).toBe(16);
		expect(degree(cfg(32, 'plain', 8))).toBe(4);
	});

	it.each(all)('every lane is served exactly once, in wavefronts 0 to degree - 1: %o', (c) => {
		const lanes = serves(c).flatMap((s) => s.lanes).sort((a, b) => a - b);
		expect(lanes).toEqual(Array.from({ length: 32 }, (_, i) => i));
		const waves = new Set(serves(c).map((s) => s.wave));
		expect(waves.size).toBe(degree(c));
		for (const a of accesses(c)) expect(a.row).toBeLessThan(64);
	});

	it.each(all)('every step has a status line, the last one says Done: %o', (c) => {
		for (let s = 0; s <= lastStep(c); s++) {
			expect(describe(c, s).length).toBeGreaterThan(40);
			expect(describe(c, s)).not.toMatch(/[\u2013\u2014]/);
		}
		expect(describe(c, lastStep(c))).toContain('Done');
		expect(stateAt(c, lastStep(c)).done).toHaveLength(serves(c).length);
	});

	it('status text for a column read', () => {
		expect(describe(cfg(32), 0)).toContain('lane 1 reads tile[1][0], word 32, bank 0');
		expect(describe(cfg(32), 2)).toBe(
			'Wavefront 2: bank 0 sends one word to lane 1. That bank already sent a different word in an earlier wavefront, so this lane had to wait. The wait is the bank conflict.',
		);
		expect(describe(cfg(32), 32)).toContain('Done: 32 wavefronts instead of 1, a 32-way bank conflict.');
		expect(describe(cfg(0), 1)).toContain('Bank 0 broadcasts word 0 to lanes 0 to 31 at once.');
		expect(describe(cfg(32, 'pad'), 1)).toContain('conflict-free');
	});

	it('lists numbers as ranges', () => {
		expect(listNumbers([3])).toBe('3');
		expect(listNumbers([2, 0, 1])).toBe('0 to 2');
		expect(listNumbers([0, 8, 16, 24])).toBe('0, 8, 16 and 24');
		expect(listNumbers(Array.from({ length: 16 }, (_, i) => 2 * i))).toBe('0, 2, 4, 6, 8, 10 and 10 more');
	});

	it('the code panel reads at the line the sim highlights', () => {
		for (const l of LAYOUT_IDS) expect(codeFor(l).split('\n')[READ_LINE - 1]).toContain('tile[r]');
		expect(codeFor('pad').split('\n')[0]).toContain('[32 + 1]');
	});
});
