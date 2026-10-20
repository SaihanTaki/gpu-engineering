import { describe as suite, expect, it } from 'vitest';
import { codeFor, COND_IDS, describe, efficiency, lastStep, program, stateAt, takesIf, X, type Config } from './model';

const both = COND_IDS.flatMap((cond) => [0, 1].map((warp) => ({ cond, warp }) as Config));

suite('divergence', () => {
	it('even lanes: both paths run, 7 instructions, 176 of 224 slots', () => {
		const c: Config = { cond: 'parity', warp: 0 };
		expect(program(c).map((x) => x.line)).toEqual([1, 2, 3, 4, 5, 7, 9]);
		const st = stateAt(c, lastStep(c));
		// 4 full instructions, 2 if instructions with 16 lanes, 1 else instruction with 16 lanes.
		expect(st.usedSlots).toBe(4 * 32 + 3 * 16);
		expect(efficiency(c)).toBeCloseTo(176 / 224, 10);
	});

	it('first 8 lanes: 8 run the if body, 24 run the else body', () => {
		const p = program({ cond: 'lower', warp: 1 });
		expect(p[3].active.filter(Boolean).length).toBe(8);
		expect(p[5].active.filter(Boolean).length).toBe(24);
	});

	it('a warp-uniform condition never diverges and skips the other path', () => {
		expect(program({ cond: 'uniform', warp: 0 }).map((x) => x.line)).toEqual([1, 2, 3, 4, 5, 9]);
		expect(program({ cond: 'uniform', warp: 1 }).map((x) => x.line)).toEqual([1, 2, 3, 7, 9]);
		expect(efficiency({ cond: 'uniform', warp: 0 })).toBe(1);
	});

	it('the data condition mixes signs in both warps', () => {
		for (const warp of [0, 1] as const) {
			const n = takesIf({ cond: 'data', warp }).filter(Boolean).length;
			expect(n).toBeGreaterThan(4);
			expect(n).toBeLessThan(28);
		}
		expect(X.every((v) => v !== 0)).toBe(true);
	});

	it.each(both)('every lane runs each statement on its path exactly once: %o', (c) => {
		const t = takesIf(c);
		for (let lane = 0; lane < 32; lane++) {
			const lines = program(c).filter((x) => x.active[lane]).map((x) => x.line);
			expect(lines).toEqual(t[lane] ? [1, 2, 3, 4, 5, 9] : [1, 2, 3, 7, 9]);
		}
	});

	it.each(both)('every step has a status line and code line: %o', (c) => {
		const code = codeFor(c.cond).split('\n');
		expect(code[2]).toContain('if (');
		for (let s = 0; s <= lastStep(c); s++) {
			expect(describe(c, s).length).toBeGreaterThan(40);
			const st = stateAt(c, s);
			if (st.instr) expect(code[st.instr.line - 1].trim().length).toBeGreaterThan(1);
		}
		expect(describe(c, lastStep(c))).toContain('Done');
	});

	it('status text for the even-lane case', () => {
		const c: Config = { cond: 'parity', warp: 0 };
		expect(describe(c, 3)).toContain('16 are true, 16 are false. The lanes disagree');
		expect(describe(c, 4)).toBe('Line 4, if path: 16 lanes run it; 16 lanes are masked off and do nothing for this instruction.');
		expect(describe(c, 7)).toContain('used 176 of them (78.6%)');
	});
});
