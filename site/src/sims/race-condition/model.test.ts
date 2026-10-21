import { describe as suite, expect, it } from 'vitest';
import {
	CODE,
	describe,
	isComplete,
	lastStep,
	lineOf,
	MODE_IDS,
	OPS,
	PRESET_IDS,
	picks,
	schedule,
	stateAt,
	THREAD_OPTIONS,
	type Config,
	type Mode,
	type Preset,
} from './model';

const cfg = (threads: number, mode: Mode, preset: Preset, custom: number[] = []): Config => ({ threads, mode, preset, custom });
const presets = PRESET_IDS.filter((p) => p !== 'custom');
const all = THREAD_OPTIONS.flatMap((t) => MODE_IDS.flatMap((m) => presets.map((p) => cfg(t, m, p))));
const final = (c: Config) => stateAt(c, lastStep(c));

suite('schedule', () => {
	it('every preset runs every instruction of every thread once', () => {
		for (const c of all) {
			const events = schedule(c);
			expect(events).toHaveLength(c.threads * OPS[c.mode].length);
			for (let t = 0; t < c.threads; t++) expect(events.filter((e) => e.thread === t).map((e) => e.op)).toEqual(OPS[c.mode]);
			expect(isComplete(c)).toBe(true);
		}
	});

	it('skips picks of finished or unknown threads', () => {
		const c = cfg(2, 'atomic', 'custom', [0, 0, 5, 1, 1]);
		expect(schedule(c)).toEqual([
			{ thread: 0, op: 'atomic' },
			{ thread: 1, op: 'atomic' },
		]);
	});

	it('the lost-one preset interleaves thread 1 inside thread 0', () => {
		expect(picks(cfg(3, 'plain', 'lost-one')).slice(0, 6)).toEqual([0, 1, 0, 0, 1, 1]);
	});
});

suite('final count', () => {
	it('serial order and atomics never lose an update', () => {
		for (const c of all.filter((c) => c.preset === 'serial' || c.mode === 'atomic')) {
			expect(final(c).count).toBe(c.threads);
			expect(final(c).lost).toBe(0);
		}
	});

	it('one interleaved load loses one update', () => {
		for (const t of THREAD_OPTIONS) {
			const s = final(cfg(t, 'plain', 'lost-one'));
			expect(s.count).toBe(t - 1);
			expect(s.lost).toBe(1);
		}
	});

	it('every thread loading first leaves count at 1', () => {
		for (const t of THREAD_OPTIONS) {
			const s = final(cfg(t, 'plain', 'all-load'));
			expect(s.count).toBe(1);
			expect(s.lost).toBe(t - 1);
		}
	});

	it('count plus lost updates is always the number of threads', () => {
		// Every order of 3 threads x 3 instructions, built from all pick sequences of length 9.
		const orders: number[][] = [];
		const grow = (seq: number[], left: number[]) => {
			if (left.every((n) => n === 0)) return void orders.push(seq);
			left.forEach((n, t) => n > 0 && grow([...seq, t], left.map((m, u) => (u === t ? m - 1 : m))));
		};
		grow([], [3, 3, 3]);
		expect(orders).toHaveLength(1680);
		const counts = new Set<number>();
		for (const o of orders) {
			const s = final(cfg(3, 'plain', 'custom', o));
			expect(s.count + s.lost).toBe(3);
			counts.add(s.count);
		}
		expect([...counts].sort()).toEqual([1, 2, 3]);
	});

	it('a late store can lower count', () => {
		// 0 loads; 1 and 2 finish; 0 stores its stale 1 over 2.
		const s = final(cfg(3, 'plain', 'custom', [0, 1, 1, 1, 2, 2, 2, 0, 0]));
		expect(s.count).toBe(1);
		expect(s.lost).toBe(2);
		expect(s.lossySteps).toEqual([8]);
	});

	it('a stale store can bring a lost update back', () => {
		// 0 loads; 1 and 2 finish; 3 loads 2; 0 stores 1 over 2; 3 stores 3.
		const c = cfg(4, 'plain', 'custom', [0, 1, 1, 1, 2, 2, 2, 3, 0, 0, 3, 3]);
		expect(stateAt(c, 10).lost).toBe(2);
		const s = final(c);
		expect(s.count).toBe(3);
		expect(s.lost).toBe(1);
		expect(s.event?.lost).toBe(-1);
		expect(describe(c, lastStep(c))).toContain('1 lost update is counted again');
	});
});

suite('state', () => {
	it('registers follow load and add', () => {
		const c = cfg(2, 'plain', 'lost-one');
		expect(stateAt(c, 0).threads.map((t) => t.reg)).toEqual([null, null]);
		expect(stateAt(c, 2).threads.map((t) => t.reg)).toEqual([0, 0]);
		expect(stateAt(c, 3).threads[0].reg).toBe(1);
		expect(stateAt(c, 4).count).toBe(1);
		const s = stateAt(c, 6);
		expect(s.event).toEqual({ thread: 1, op: 'store', before: 1, after: 1, lost: 1 });
	});

	it('steps past the end clamp to the last state', () => {
		const c = cfg(2, 'plain', 'serial');
		expect(stateAt(c, 99)).toEqual(stateAt(c, lastStep(c)));
	});
});

suite('describe', () => {
	it('has a status for every step', () => {
		for (const c of all) for (let s = 0; s <= lastStep(c); s++) expect(describe(c, s).length).toBeGreaterThan(20);
	});

	it('reports lost updates at the end', () => {
		const c = cfg(4, 'plain', 'all-load');
		expect(describe(c, lastStep(c))).toContain('3 updates lost');
		const a = cfg(4, 'atomic', 'all-load');
		expect(describe(a, lastStep(a))).toContain('no update lost');
	});

	it('asks the reader to pick in custom mode', () => {
		expect(describe(cfg(2, 'plain', 'custom'), 0)).toContain('Pick which thread');
	});
});

suite('code panel', () => {
	it('every instruction maps to a line of its code', () => {
		for (const m of MODE_IDS) {
			const lines = CODE[m].split('\n');
			for (const op of OPS[m]) expect(lines[lineOf(op) - 1]).toMatch(op === 'atomic' ? /atomicAdd/ : op === 'load' ? /r = count/ : op === 'add' ? /r = r \+ 1/ : /count = r/);
		}
	});
});
