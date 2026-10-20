import { describe as suite, expect, it } from 'vitest';
import {
	amdahl,
	describe,
	endTime,
	laneAt,
	laneById,
	lanes,
	lastStep,
	modelSpeedup,
	STEPS,
	timeAt,
	winner,
	type Config,
} from './model';

const base: Config = { p: 90, work: 10000, cpuWorkers: 8, gpuWorkers: 1000 };
const at = (p: number, c: Partial<Config> = {}) => ({ ...base, ...c, p });
const finishes = (c: Config) => lanes(c).map((l) => l.finish);

suite("Amdahl's law", () => {
	it('matches hand-computed values', () => {
		expect(amdahl(0.5, 8)).toBeCloseTo(1 / 0.5625, 10);
		expect(amdahl(0.9, 8)).toBeCloseTo(4.70588, 4);
		expect(amdahl(0.9, 1000)).toBeCloseTo(9.91080, 4);
		expect(amdahl(0.99, 1000)).toBeCloseTo(90.99181, 4);
		expect(amdahl(0.95, 8)).toBeCloseTo(5.92593, 4);
		expect(amdahl(0.95, 1000)).toBeCloseTo(19.62709, 4);
	});

	it('gives 1 with no parallel part and n with no serial part', () => {
		expect(amdahl(0, 1000)).toBe(1);
		expect(amdahl(1, 1000)).toBeCloseTo(1000, 9);
		expect(amdahl(0.7, 1)).toBeCloseTo(1, 12);
	});

	it('never passes 1 / (1 - p)', () => {
		expect(amdahl(0.8, 1e9)).toBeLessThan(5);
		expect(amdahl(0.8, 1e9)).toBeCloseTo(5, 6);
	});

	it('the model speedup is Amdahl when the parallel units divide evenly', () => {
		const c = at(90, { work: 8000 });
		expect(modelSpeedup(laneById(c, 'cpu'))).toBeCloseTo(amdahl(0.9, 8), 10);
		expect(modelSpeedup(laneById(at(90), 'gpu'))).toBeCloseTo(40000 / 4036, 10);
	});
});

suite('race finish times with the defaults', () => {
	it('p = 90%: the CPU wins', () => {
		expect(finishes(at(90))).toEqual([2125, 4036, 1036]);
		expect(winner(at(90))).toBe('cpu');
	});

	it('p = 99%: the many slow workers win', () => {
		// CPU: 100 + ceil(9900 / 8) = 1338. GPU: 4 * 100 + 4 * ceil(9900 / 1000) = 440.
		expect(finishes(at(99))).toEqual([1338, 440, 140]);
		expect(winner(at(99))).toBe('gpu');
	});

	it('p = 50%: the few fast workers win', () => {
		expect(finishes(at(50))).toEqual([5625, 20020, 5020]);
		expect(winner(at(50))).toBe('cpu');
	});

	it('the crossover is between 96% and 97%', () => {
		expect(finishes(at(96)).slice(0, 2)).toEqual([1600, 1640]);
		expect(finishes(at(97)).slice(0, 2)).toEqual([1513, 1240]);
		for (let p = 0; p <= 100; p++) expect(winner(at(p)), `p = ${p}`).toBe(p <= 96 ? 'cpu' : 'gpu');
	});

	it('serial on the CPU and parallel on the GPU is never slower than either', () => {
		for (let p = 0; p <= 100; p++) {
			const [cpu, gpu, both] = finishes(at(p));
			expect(both).toBeLessThanOrEqual(Math.min(cpu, gpu));
		}
	});

	it('handles p = 0 and p = 100', () => {
		expect(finishes(at(0))).toEqual([10000, 40000, 10000]);
		expect(finishes(at(100))).toEqual([1250, 40, 40]);
	});

	it('a job smaller than the GPU leaves workers idle', () => {
		const c = at(99, { work: 1000 });
		const gpu = laneById(c, 'gpu');
		expect(gpu.finish).toBe(44);
		expect(laneAt(gpu, 40).busy).toBe(990);
	});
});

suite('stepping', () => {
	it('runs from t = 0 to the slowest finish in STEPS steps', () => {
		expect(lastStep).toBe(STEPS);
		expect(timeAt(base, 0)).toBe(0);
		expect(timeAt(base, STEPS)).toBe(endTime(base));
		expect(timeAt(base, STEPS / 2)).toBe(endTime(base) / 2);
	});

	it('clamps steps outside the range', () => {
		expect(timeAt(base, -3)).toBe(0);
		expect(timeAt(base, STEPS + 5)).toBe(endTime(base));
		expect(describe(base, STEPS + 5)).toBe(describe(base, STEPS));
	});

	it('every lane starts empty and ends finished', () => {
		for (const l of lanes(base)) {
			expect(laneAt(l, 0)).toMatchObject({ phase: 'waiting', fraction: 0, busy: 0 });
			expect(laneAt(l, timeAt(base, STEPS))).toMatchObject({ phase: 'done', fraction: 1, busy: 0 });
		}
	});

	it('the serial part uses one worker, the parallel part all of them', () => {
		const gpu = laneById(base, 'gpu');
		expect(laneAt(gpu, 100)).toMatchObject({ phase: 'serial', serialDone: 25, busy: 1 });
		expect(laneAt(gpu, 4001)).toMatchObject({ phase: 'parallel', parallelDone: 0, busy: 1000 });
		expect(laneAt(gpu, 4032)).toMatchObject({ phase: 'parallel', parallelDone: 8000, busy: 1000 });
		const cpu = laneById(base, 'cpu');
		// 9000 units over 8 workers: 1125 full rounds, so the last round is full too.
		expect(laneAt(cpu, 2124)).toMatchObject({ phase: 'parallel', parallelDone: 8992, busy: 8 });
	});

	it('progress never goes backwards', () => {
		for (const l of lanes(base)) {
			let prev = -1;
			for (let s = 0; s <= STEPS; s++) {
				const f = laneAt(l, timeAt(base, s)).fraction;
				expect(f).toBeGreaterThanOrEqual(prev);
				prev = f;
			}
		}
	});

	it('describes the start, a middle step and the result', () => {
		expect(describe(base, 0)).toContain('Predict which finishes first');
		expect(describe(base, 20)).toMatch(/^t = 2,018 ticks\. CPU 91%, parallel, 8 of 8 busy\. GPU 5%, serial, 1 worker busy\./);
		expect(describe(base, STEPS)).toContain('The CPU wins, 2,125 ticks against 4,036.');
		expect(describe(at(99), STEPS)).toContain('The GPU wins, 440 ticks against 1,338.');
		expect(describe(at(99), STEPS)).toContain('takes 140.');
	});
});
