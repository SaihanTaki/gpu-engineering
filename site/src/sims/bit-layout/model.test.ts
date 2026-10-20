import { describe as suite, expect, it } from 'vitest';
import {
	bitString,
	convert,
	decode,
	describe,
	flipBit,
	formatById,
	FORMATS,
	hex,
	maxFinite,
	minNormal,
	minSubnormal,
	parseValue,
	TOUR,
	totalBits,
	type Bits,
} from './model';

const f32 = formatById('fp32');
const tf32 = formatById('tf32');
const f16 = formatById('fp16');
const bf16 = formatById('bf16');
const e4m3 = formatById('e4m3');
const e5m2 = formatById('e5m2');

/** FP16 bits of an FP32 value by integer shifts on its bit pattern, independent of convert(). */
function f16Reference(x: number) {
	const buf = new DataView(new ArrayBuffer(4));
	buf.setFloat32(0, x);
	const w = buf.getUint32(0);
	const sign = w >>> 31;
	let exp = ((w >>> 23) & 0xff) - 127 + 15;
	let man = w & 0x7fffff;
	if (exp >= 31) return (sign << 15) | 0x7c00;
	if (exp <= 0) {
		// Subnormal: shift the implicit 1 in and round to nearest even.
		if (exp < -10) return sign << 15;
		man |= 0x800000;
		const shift = 14 - exp;
		const half = 1 << (shift - 1);
		const rest = man & ((1 << shift) - 1);
		let m = man >>> shift;
		if (rest > half || (rest === half && m & 1)) m++;
		return (sign << 15) | m;
	}
	const rest = man & 0x1fff;
	man >>>= 13;
	if (rest > 0x1000 || (rest === 0x1000 && man & 1)) {
		man++;
		if (man === 0x400) {
			man = 0;
			exp++;
			if (exp >= 31) return (sign << 15) | 0x7c00;
		}
	}
	return (sign << 15) | (exp << 10) | man;
}

suite('format limits', () => {
	it('largest finite values', () => {
		expect(maxFinite(f16)).toBe(65504);
		expect(maxFinite(e4m3)).toBe(448);
		expect(maxFinite(e5m2)).toBe(57344);
		expect(maxFinite(bf16)).toBe((2 - 2 ** -7) * 2 ** 127);
		expect(maxFinite(f32)).toBe(3.4028234663852886e38);
		expect(maxFinite(tf32)).toBe((2 - 2 ** -10) * 2 ** 127);
	});

	it('smallest normal and subnormal values', () => {
		expect(minNormal(f16)).toBe(2 ** -14);
		expect(minSubnormal(f16)).toBe(2 ** -24);
		expect(minNormal(e4m3)).toBe(2 ** -6);
		expect(minSubnormal(e4m3)).toBe(2 ** -9);
		expect(minNormal(e5m2)).toBe(2 ** -14);
		expect(minSubnormal(e5m2)).toBe(2 ** -16);
		expect(minSubnormal(f32)).toBe(2 ** -149);
		expect(minSubnormal(bf16)).toBe(2 ** -133);
	});

	it('bit widths', () => {
		expect(FORMATS.map(totalBits)).toEqual([32, 19, 16, 16, 8, 8]);
	});
});

suite('round to nearest even', () => {
	it('FP16 known encodings', () => {
		expect(hex(f16, convert(f16, 1))).toBe('0x3C00');
		expect(hex(f16, convert(f16, 65504))).toBe('0x7BFF');
		expect(hex(f16, convert(f16, 1 / 3))).toBe('0x3555');
		expect(convert(f16, 1 / 3).value).toBe(0.333251953125);
		expect(hex(f16, convert(f16, -2))).toBe('0xC000');
		expect(hex(f16, convert(f16, 2 ** -24))).toBe('0x0001');
		expect(hex(f16, convert(f16, 0.1))).toBe('0x2E66');
	});

	it('FP16 overflow and underflow at the ties', () => {
		// 65520 is halfway between 65504 (odd mantissa) and 65536: ties to even goes up, to inf.
		expect(convert(f16, 65520).value).toBe(Infinity);
		expect(convert(f16, 65520).overflow).toBe(true);
		expect(convert(f16, 65519.99).value).toBe(65504);
		// 2^-25 is half the smallest subnormal; even neighbour is 0.
		expect(convert(f16, 2 ** -25).value).toBe(0);
		expect(convert(f16, 2 ** -25).underflow).toBe(true);
		expect(convert(f16, 2 ** -25 * 1.0001).value).toBe(2 ** -24);
		expect(convert(f16, 3 * 2 ** -25).value).toBe(2 ** -23);
		expect(convert(f16, 1e-5).kind).toBe('subnormal');
	});

	it('FP16 matches an independent integer reference on many values', () => {
		let seed = 7;
		const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
		for (let i = 0; i < 20000; i++) {
			const x = Math.fround((rand() - 0.5) * 2 ** (Math.floor(rand() * 50) - 30));
			const got = convert(f16, x);
			const word = (got.sign << 15) | (got.exp << 10) | got.man;
			expect(word).toBe(f16Reference(x));
		}
	});

	it('FP32 matches Math.fround', () => {
		let seed = 3;
		const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
		for (let i = 0; i < 20000; i++) {
			const x = (rand() - 0.5) * 2 ** (Math.floor(rand() * 300) - 150);
			expect(convert(f32, x).value).toBe(Math.fround(x));
		}
		expect(convert(f32, 1e39).value).toBe(Infinity);
		expect(convert(f32, 2 ** -150).value).toBe(0);
	});

	it('BF16 of 1/3 is 0x3EAB', () => {
		const c = convert(bf16, 1 / 3);
		expect(hex(bf16, c)).toBe('0x3EAB');
		expect(c.value).toBe(0.333984375);
		expect(bitString(bf16, c)).toBe('0 01111101 0101011');
		// BF16 is the top half of FP32: FP32 1/3 is 0x3EAAAAAB, which rounds up to 0x3EAB.
		expect(hex(f32, convert(f32, 1 / 3))).toBe('0x3EAAAAAB');
	});

	it('BF16 ties to even', () => {
		// 1 + 2^-8 is halfway between 1 and 1 + 2^-7: even mantissa is 0.
		expect(convert(bf16, 1 + 2 ** -8).value).toBe(1);
		expect(convert(bf16, 1 + 3 * 2 ** -8).value).toBe(1 + 2 * 2 ** -7);
		expect(convert(bf16, 65504).value).toBe(65536);
	});

	it('TF32 keeps 10 mantissa bits and FP32 range', () => {
		expect(convert(tf32, 1 / 3).value).toBe(convert(f16, 1 / 3).value);
		expect(convert(tf32, 3e38).kind).toBe('normal');
		expect(convert(tf32, 1 + 2 ** -11).value).toBe(1);
		expect(convert(tf32, 1 + 3 * 2 ** -11).value).toBe(1 + 2 ** -9);
	});

	it('E4M3: max 448, no infinity, NaN past the range', () => {
		const c = convert(e4m3, 448);
		expect(bitString(e4m3, c)).toBe('0 1111 110');
		expect(c.value).toBe(448);
		// 464 is the tie between 448 (mantissa 110, even) and 480 (the NaN slot); ties to even keeps 448.
		expect(convert(e4m3, 464).value).toBe(448);
		expect(Number.isNaN(convert(e4m3, 464.01).value)).toBe(true);
		expect(convert(e4m3, 500).overflow).toBe(true);
		expect(convert(e4m3, Infinity).kind).toBe('nan');
		expect(convert(e4m3, 1 / 3).value).toBe(0.34375);
		expect(convert(e4m3, 2 ** -9).kind).toBe('subnormal');
		expect(convert(e4m3, 2 ** -10).value).toBe(0);
	});

	it('E5M2: max 57344, IEEE inf', () => {
		expect(bitString(e5m2, convert(e5m2, 57344))).toBe('0 11110 11');
		expect(convert(e5m2, 61440).value).toBe(Infinity);
		expect(convert(e5m2, 61439).value).toBe(57344);
		expect(convert(e5m2, 1 / 3).value).toBe(0.3125);
		expect(convert(e5m2, 2 ** -16).kind).toBe('subnormal');
	});

	it('errors and signs', () => {
		const c = convert(f16, 0.1);
		expect(c.absError).toBeCloseTo(Math.abs(0.0999755859375 - 0.1), 15);
		expect(c.relError).toBeLessThanOrEqual(2 ** -11);
		expect(convert(f16, -0).sign).toBe(1);
		expect(convert(f16, -0).relError).toBeNaN();
		expect(convert(f16, NaN).kind).toBe('nan');
		expect(convert(f16, -Infinity).value).toBe(-Infinity);
	});

	it('round trips every bit pattern of each 8- and 16-bit format', () => {
		for (const f of [f16, bf16, e4m3, e5m2]) {
			for (let w = 0; w < 2 ** totalBits(f); w++) {
				const b: Bits = {
					sign: w >>> (totalBits(f) - 1),
					exp: (w >>> f.manBits) & ((1 << f.expBits) - 1),
					man: w & ((1 << f.manBits) - 1),
				};
				const d = decode(f, b);
				if (d.kind === 'nan') continue;
				const c = convert(f, d.value);
				expect([c.sign, c.exp, c.man]).toEqual([b.sign, b.exp, b.man]);
			}
		}
	});
});

suite('bit editing', () => {
	it('flipping the sign and the top exponent bit', () => {
		const one = convert(f16, 1);
		expect(decode(f16, flipBit(f16, one, 0)).value).toBe(-1);
		// 1.0 is 0 01111 0000000000; setting the top exponent bit gives exponent field 11111: inf.
		expect(decode(f16, flipBit(f16, one, 1)).kind).toBe('inf');
		// Lowest mantissa bit: the next number after 1.
		expect(decode(f16, flipBit(f16, one, 15)).value).toBe(1 + 2 ** -10);
		expect(decode(f32, flipBit(f32, convert(f32, 1), 31)).value).toBe(1 + 2 ** -23);
	});
});

suite('text', () => {
	it('parses numbers', () => {
		expect(parseValue('1e-5')).toBe(1e-5);
		expect(parseValue(' -2^-25 ')).toBe(-(2 ** -25));
		expect(parseValue('inf')).toBe(Infinity);
		expect(parseValue('.5')).toBe(0.5);
		expect(Number.isNaN(parseValue('nan'))).toBe(true);
		expect(parseValue('abc')).toBeNull();
		expect(parseValue('')).toBeNull();
	});

	it('tour stops say what the model computes', () => {
		const at = (label: string) => TOUR.find((t) => t.label === label)!.value;
		expect(convert(f16, at('65520')).kind).toBe('inf');
		expect(convert(bf16, at('65520')).value).toBe(65536);
		expect(convert(f16, at('1e-5')).kind).toBe('subnormal');
		expect(convert(f16, at('2^-25')).value).toBe(0);
		expect(convert(bf16, at('2^-25')).value).toBe(2 ** -25);
		expect(convert(f16, at('3e38')).value).toBe(Infinity);
		for (const id of ['fp32', 'tf32', 'bf16'] as const) expect(convert(formatById(id), at('3e38')).kind).toBe('normal');
		expect(Number.isNaN(convert(e4m3, at('500')).value)).toBe(true);
		const rel = (id: string) => convert(formatById(id), 1 / 3).relError;
		expect(rel('fp32')).toBeCloseTo(3e-8, 8);
		expect(rel('fp16')).toBeCloseTo(2.4e-4, 5);
		expect(rel('bf16')).toBeCloseTo(2e-3, 3);
		expect(rel('e4m3')).toBeCloseTo(3e-2, 2);
		for (const id of ['fp16', 'tf32']) expect(convert(formatById(id), 0.1).relError).toBeLessThanOrEqual(2 ** -11);
		expect(convert(bf16, 0.1).relError).toBeLessThanOrEqual(2 ** -8);
		expect(convert(e5m2, 0.1).relError).toBeLessThanOrEqual(2 ** -3);
		expect(convert(bf16, 65504).absError).toBe(32);
	});

	it('describe follows the step and the value', () => {
		expect(describe({ value: 1, edit: 'fp16' }, 0)).toMatch(/^Step 0/);
		expect(describe({ value: TOUR[0].value, edit: 'fp16' }, 1)).toMatch(/^Step 1, x = 1\/3/);
		expect(describe({ value: 2, edit: 'fp16' }, 1)).toMatch(/^x = 2: FP32 2 \(exact\)/);
	});
});
