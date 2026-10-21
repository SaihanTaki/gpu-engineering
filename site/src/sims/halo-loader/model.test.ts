import { describe as suite, expect, it } from 'vitest';
import {
	CODE,
	describe,
	haloCells,
	inImage,
	lastStep,
	overhead,
	POSITIONS,
	RADIUS_RANGE,
	readers,
	rounds,
	roundCounts,
	side,
	stateAt,
	steps,
	TILE_RANGE,
	totals,
	type Config,
} from './model';

const range = (r: { min: number; max: number; step: number }) => {
	const out: number[] = [];
	for (let v = r.min; v <= r.max; v += r.step) out.push(v);
	return out;
};
const all = POSITIONS.flatMap((position) =>
	range(TILE_RANGE).flatMap((tile) => range(RADIUS_RANGE).map((radius) => ({ tile, radius, position }) as Config)),
);

/** Naive loads by brute force: every output's window, in-image pixels only. */
function naiveBrute(c: Config) {
	let n = 0;
	for (let y = 0; y < c.tile; y++)
		for (let x = 0; x < c.tile; x++)
			for (let dy = 0; dy <= 2 * c.radius; dy++)
				for (let dx = 0; dx <= 2 * c.radius; dx++) if (inImage(c, y + dy, x + dx)) n++;
	return n;
}

suite('halo loader', () => {
	it('T = 16, R = 2 inside the image: 20 x 20 tile, 144 halo cells, 1.5625 loads per output against 25', () => {
		const c: Config = { tile: 16, radius: 2, position: 'inside' };
		expect(side(c)).toBe(20);
		expect(haloCells(c)).toBe(144);
		expect(overhead(c)).toBe(0.5625);
		expect(rounds(c)).toBe(2);
		const t = totals(c);
		expect(t.tiled).toBe(400);
		expect(t.naive).toBe(256 * 25);
		expect(t.tiledPerOutput).toBe(1.5625);
		expect(t.naivePerOutput).toBe(25);
		expect(t.naive / t.tiled).toBe(16);
	});

	it('the halo outgrows the tile once R > (sqrt 2 - 1) T / 2', () => {
		expect(overhead({ tile: 8, radius: 1, position: 'inside' })).toBeLessThan(1);
		expect(overhead({ tile: 8, radius: 2, position: 'inside' })).toBeGreaterThan(1);
		expect(overhead({ tile: 16, radius: 3, position: 'inside' })).toBeLessThan(1);
		expect(overhead({ tile: 16, radius: 4, position: 'inside' })).toBeGreaterThan(1);
	});

	it.each(all)('closed forms hold: %o', (c) => {
		const t = totals(c);
		const k = 2 * c.radius + 1;
		// Every output reads its whole window from shared memory, zeros included.
		expect(t.sharedReads).toBe(c.tile ** 2 * k * k);
		expect(t.naive).toBe(naiveBrute(c));
		if (c.position === 'inside') {
			expect(t.tiled).toBe(side(c) ** 2);
			expect(t.naive).toBe(t.sharedReads);
			expect(t.zeros).toBe(0);
		} else {
			// The top R rows and left R columns are above and left of the image.
			expect(t.tiled).toBe((c.tile + c.radius) ** 2);
		}
		expect(t.tiled + t.zeros).toBe(side(c) ** 2);
	});

	it.each(all)('the final state has loaded every in-image slot once: %o', (c) => {
		const end = stateAt(c, lastStep(c));
		const t = totals(c);
		expect(end.loads).toBe(t.tiled);
		expect(end.zeros).toBe(t.zeros);
		expect(end.stored).toBe(true);
		let sum = 0;
		for (let k = 0; k < rounds(c); k++) sum += roundCounts(c, k).loads;
		expect(sum).toBe(t.tiled);
	});

	it('readers: corners of the tile are read by one thread, the centre by (2R+1)^2', () => {
		const c: Config = { tile: 8, radius: 1, position: 'inside' };
		expect(readers(c, 0, 0)).toBe(1);
		expect(readers(c, 9, 9)).toBe(1);
		expect(readers(c, 4, 4)).toBe(9);
		expect(readers(c, 0, 4)).toBe(3);
	});

	it('the last round is partial when T^2 does not divide (T + 2R)^2', () => {
		const c: Config = { tile: 16, radius: 2, position: 'inside' };
		expect(roundCounts(c, 0)).toMatchObject({ active: 256, loads: 256 });
		expect(roundCounts(c, 1)).toMatchObject({ first: 256, end: 400, active: 144 });
		expect(describe(c, 2)).toContain('threads 0 to 143 fill slots 256 to 399; the other 112 have nothing left to load');
	});

	it('step counts: rounds + 4 states', () => {
		expect(steps({ tile: 16, radius: 2, position: 'inside' }).length).toBe(6);
		expect(lastStep({ tile: 4, radius: 8, position: 'inside' })).toBe(25 + 3);
	});

	it.each(all)('every step has a status and valid code lines: %o', (c) => {
		const code = CODE.split('\n');
		for (let s = 0; s <= lastStep(c); s++) {
			expect(describe(c, s).length).toBeGreaterThan(40);
			for (const line of stateAt(c, s).cur.lines) expect(code[line - 1].trim().length).toBeGreaterThan(1);
		}
		expect(describe(c, lastStep(c))).toContain('Done');
	});

	it('code lines point at the statements they describe', () => {
		const code = CODE.split('\n');
		const kinds = Object.fromEntries(steps({ tile: 8, radius: 1, position: 'inside' }).map((x) => [x.kind, x.lines]));
		expect(code[kinds.load[0] - 1]).toContain('for (int i = threadIdx.y * TILE');
		expect(code[kinds.load[3] - 1]).toContain('inside ? in[yy * w + xx] : 0.0f');
		expect(code[kinds.sync[0] - 1]).toContain('__syncthreads');
		expect(code[kinds.compute[1] - 1]).toContain('if (x >= w || y >= h) return;');
		expect(code[kinds.compute[5] - 1]).toContain('c_filter[dy * (2 * R + 1) + dx] * tile');
		expect(code[kinds.store[0] - 1]).toContain('out[');
	});

	it('status text at the corner counts the zero padding', () => {
		const c: Config = { tile: 4, radius: 1, position: 'corner' };
		// 6 x 6 slots in rounds of 16. Round 1 is rows 0 and 1 and four slots of row 2; row 0 and column 0 are outside.
		expect(describe(c, 1)).toBe(
			'Round 1 of 3: every thread fills one slot, i = 0 to 15. 8 global loads, 1 of them halo pixels. 8 slots lie outside the image and get 0 with no load. Loaded so far: 8 of 25.',
		);
		expect(describe(c, lastStep(c))).toContain('25 global loads for 16 outputs');
	});
});
