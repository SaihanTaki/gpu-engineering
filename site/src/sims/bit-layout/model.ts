/**
 * Floating-point formats used by tensor cores, with exact round-to-nearest-even
 * conversion from a JavaScript number (an IEEE double) and bit-level decoding.
 *
 * Every format here has fewer bits than a double, so each rounding step below
 * is exact in double arithmetic: scaling by powers of two and floor() lose nothing.
 */

export type FormatId = 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'e4m3' | 'e5m2';

export interface Format {
	id: FormatId;
	name: string;
	expBits: number;
	manBits: number;
	bias: number;
	/** IEEE style: all-ones exponent means inf (mantissa 0) or NaN. E4M3 has no inf. */
	ieeeSpecials: boolean;
	note: string;
}

export const FORMATS: Format[] = [
	{ id: 'fp32', name: 'FP32', expBits: 8, manBits: 23, bias: 127, ieeeSpecials: true, note: 'IEEE 754 binary32' },
	{ id: 'tf32', name: 'TF32', expBits: 8, manBits: 10, bias: 127, ieeeSpecials: true, note: 'tensor-core input format, stored in 32 bits' },
	{ id: 'fp16', name: 'FP16', expBits: 5, manBits: 10, bias: 15, ieeeSpecials: true, note: 'IEEE 754 binary16' },
	{ id: 'bf16', name: 'BF16', expBits: 8, manBits: 7, bias: 127, ieeeSpecials: true, note: 'top 16 bits of FP32' },
	{ id: 'e4m3', name: 'FP8 E4M3', expBits: 4, manBits: 3, bias: 7, ieeeSpecials: false, note: 'no infinity; S.1111.111 is NaN' },
	{ id: 'e5m2', name: 'FP8 E5M2', expBits: 5, manBits: 2, bias: 15, ieeeSpecials: true, note: 'IEEE style specials' },
];

export const FORMAT_IDS = FORMATS.map((f) => f.id);
export const formatById = (id: string) => FORMATS.find((f) => f.id === id) ?? FORMATS[0];
export const totalBits = (f: Format) => 1 + f.expBits + f.manBits;

/** Exponent of the smallest normal number. */
export const minNormalExp = (f: Format) => 1 - f.bias;
export const minSubnormal = (f: Format) => 2 ** (minNormalExp(f) - f.manBits);
export const minNormal = (f: Format) => 2 ** minNormalExp(f);
export function maxFinite(f: Format) {
	const maxField = (1 << f.expBits) - (f.ieeeSpecials ? 2 : 1);
	// E4M3 gives up the all-ones mantissa at the top exponent to NaN.
	const maxMan = f.ieeeSpecials ? 2 ** f.manBits - 1 : 2 ** f.manBits - 2;
	return (1 + maxMan / 2 ** f.manBits) * 2 ** (maxField - f.bias);
}

export type Kind = 'zero' | 'subnormal' | 'normal' | 'inf' | 'nan';

export interface Bits {
	sign: number;
	exp: number;
	man: number;
}

export interface Decoded extends Bits {
	value: number;
	kind: Kind;
}

/** floor(log2(x)) for finite x > 0, exact. */
function ilog2(x: number) {
	let e = Math.floor(Math.log2(x));
	while (2 ** e > x) e--;
	while (2 ** (e + 1) <= x) e++;
	return e;
}

function roundHalfEven(v: number) {
	const f = Math.floor(v);
	const d = v - f;
	if (d > 0.5) return f + 1;
	if (d < 0.5) return f;
	return f % 2 === 0 ? f : f + 1;
}

export function decode(f: Format, b: Bits): Decoded {
	const top = (1 << f.expBits) - 1;
	const s = b.sign ? -1 : 1;
	const frac = b.man / 2 ** f.manBits;
	if (f.ieeeSpecials && b.exp === top) return { ...b, value: b.man === 0 ? s * Infinity : NaN, kind: b.man === 0 ? 'inf' : 'nan' };
	if (!f.ieeeSpecials && b.exp === top && b.man === 2 ** f.manBits - 1) return { ...b, value: NaN, kind: 'nan' };
	if (b.exp === 0) {
		const value = s * frac * minNormal(f);
		return { ...b, value, kind: b.man === 0 ? 'zero' : 'subnormal' };
	}
	return { ...b, value: s * (1 + frac) * 2 ** (b.exp - f.bias), kind: 'normal' };
}

/** Bits of the canonical NaN: all-ones exponent, top mantissa bit set (E4M3: all ones). */
const nanBits = (f: Format, sign = 0): Bits => ({
	sign,
	exp: (1 << f.expBits) - 1,
	man: f.ieeeSpecials ? 2 ** (f.manBits - 1) : 2 ** f.manBits - 1,
});

export interface Conversion extends Decoded {
	input: number;
	/** |rounded - input|; NaN when either is not finite. */
	absError: number;
	/** absError / |input|; NaN for input 0 or non-finite values. */
	relError: number;
	overflow: boolean;
	/** A nonzero input that rounded to zero. */
	underflow: boolean;
}

/**
 * Round x to the format with round-to-nearest, ties to even, as IEEE 754 and
 * the CUDA __float2half_rn / __float2bfloat16_rn conversions do. Values past
 * the largest finite number become inf, or NaN in E4M3 (the non-saturating
 * conversion; CUDA's __NV_SATFINITE mode clamps to 448 instead).
 */
export function convert(f: Format, x: number): Conversion {
	const sign = x < 0 || Object.is(x, -0) ? 1 : 0;
	const out = (d: Decoded, overflow = false): Conversion => {
		const finite = Number.isFinite(x) && Number.isFinite(d.value);
		const absError = finite ? Math.abs(d.value - x) : NaN;
		return {
			...d,
			input: x,
			absError,
			relError: finite && x !== 0 ? absError / Math.abs(x) : NaN,
			overflow,
			underflow: x !== 0 && Number.isFinite(x) && d.kind === 'zero',
		};
	};
	if (Number.isNaN(x)) return out(decode(f, nanBits(f)));
	const top = (1 << f.expBits) - 1;
	const a = Math.abs(x);
	if (a === Infinity) return out(f.ieeeSpecials ? decode(f, { sign, exp: top, man: 0 }) : decode(f, nanBits(f, sign)));
	if (a === 0) return out(decode(f, { sign, exp: 0, man: 0 }));
	// The spacing between neighbours in a's binade; below the normal range it stays at the subnormal spacing.
	const e = Math.max(ilog2(a), minNormalExp(f));
	const q = 2 ** (e - f.manBits);
	const r = roundHalfEven(a / q) * q;
	if (r > maxFinite(f)) {
		return out(f.ieeeSpecials ? decode(f, { sign, exp: top, man: 0 }) : decode(f, nanBits(f, sign)), true);
	}
	return out(decode(f, encode(f, sign, r)));
}

/** Bits of a value r >= 0 that the format represents exactly. */
function encode(f: Format, sign: number, r: number): Bits {
	if (r === 0) return { sign, exp: 0, man: 0 };
	if (r < minNormal(f)) return { sign, exp: 0, man: r / minSubnormal(f) };
	const e = ilog2(r);
	return { sign, exp: e + f.bias, man: (r / 2 ** e - 1) * 2 ** f.manBits };
}

/** The bits as one unsigned integer, sign first. */
export const toWord = (f: Format, b: Bits) => (b.sign * 2 ** (f.expBits + f.manBits) + b.exp * 2 ** f.manBits + b.man) >>> 0;

/** Bit i, counted from the most significant (0 = sign). */
export function bitAt(f: Format, b: Bits, i: number) {
	const w = toWord(f, b);
	return Math.floor(w / 2 ** (totalBits(f) - 1 - i)) % 2;
}

export function flipBit(f: Format, b: Bits, i: number): Bits {
	const n = totalBits(f);
	const w = toWord(f, b);
	const mask = 2 ** (n - 1 - i);
	const v = bitAt(f, b, i) ? w - mask : w + mask;
	const manMod = 2 ** f.manBits;
	return {
		sign: Math.floor(v / 2 ** (n - 1)) % 2,
		exp: Math.floor(v / manMod) % 2 ** f.expBits,
		man: v % manMod,
	};
}

export const fieldOf = (f: Format, i: number): 'sign' | 'exp' | 'man' => (i === 0 ? 'sign' : i <= f.expBits ? 'exp' : 'man');

export function bitString(f: Format, b: Bits) {
	const s = Array.from({ length: totalBits(f) }, (_, i) => String(bitAt(f, b, i))).join('');
	return `${s[0]} ${s.slice(1, 1 + f.expBits)} ${s.slice(1 + f.expBits)}`;
}

export function hex(f: Format, b: Bits) {
	return '0x' + toWord(f, b).toString(16).toUpperCase().padStart(Math.ceil(totalBits(f) / 4), '0');
}

/** Shortest decimal that reads back as the same double, capped for display. */
export function fmtValue(v: number) {
	if (Number.isNaN(v)) return 'NaN';
	if (v === Infinity) return 'inf';
	if (v === -Infinity) return '-inf';
	if (v === 0) return Object.is(v, -0) ? '-0' : '0';
	const a = Math.abs(v);
	if (a >= 1e-4 && a < 1e9) return String(Number(v.toPrecision(12)));
	return v.toExponential(8).replace(/\.?0+e/, 'e');
}

export function fmtError(v: number) {
	if (Number.isNaN(v)) return 'n/a';
	if (v === 0) return '0';
	return v.toExponential(2);
}

/** The formula with this value's numbers filled in. */
export function formula(f: Format, d: Decoded) {
	const s = d.sign ? '-' : '+';
	if (d.kind === 'nan') return 'NaN';
	if (d.kind === 'inf') return `${s}inf`;
	if (d.kind === 'zero') return `${s}0`;
	if (d.kind === 'subnormal') return `${s}0.${d.man.toString(2).padStart(f.manBits, '0')} (binary) x 2^${minNormalExp(f)}`;
	return `${s}1.${d.man.toString(2).padStart(f.manBits, '0')} (binary) x 2^${d.exp - f.bias}`;
}

export function kindText(c: Conversion) {
	if (c.overflow) return c.kind === 'inf' ? 'overflow to inf' : 'overflow to NaN';
	if (c.underflow) return 'underflow to 0';
	if (c.kind === 'subnormal') return 'subnormal';
	if (c.kind === 'zero') return 'zero';
	if (c.kind === 'inf') return 'infinity';
	if (c.kind === 'nan') return 'NaN';
	return c.absError === 0 ? 'exact' : 'normal, rounded';
}

/** Tour stops for the stepper. Step k (k >= 1) shows TOUR[k - 1]. */
export const TOUR: { value: number; label: string; text: string }[] = [
	{
		value: 1 / 3,
		label: '1/3',
		text: '1/3 has an endless binary expansion 0.010101..., so every format rounds it. Fewer mantissa bits, larger error: the relative error is about 3e-8 in FP32, 2e-4 in FP16, 2e-3 in BF16 and 3e-2 in E4M3.',
	},
	{
		value: 0.1,
		label: '0.1',
		text: '0.1 is not exact in any binary format. Its relative error is at most 2^-(m+1) for m mantissa bits: 2^-11 for FP16 and TF32, 2^-8 for BF16, 2^-3 for E5M2.',
	},
	{
		value: 65504,
		label: '65504',
		text: '65504 is the largest finite FP16 number: exponent field 11110, mantissa all ones. BF16, which keeps FP32\'s 8 exponent bits, stores it with an error of 32 but has range left over.',
	},
	{
		value: 65520,
		label: '65520',
		text: '65520 lies exactly halfway between 65504 and the next step, 65536, which FP16 cannot hold. Ties go to the even mantissa, which is the one above, so FP16 overflows to inf. BF16 rounds it to 65536.',
	},
	{
		value: 448,
		label: '448',
		text: '448 is the largest E4M3 number. E4M3 has no infinity: it uses the top exponent for numbers and keeps only S.1111.111 as NaN. With IEEE rules for specials the largest value would be 240.',
	},
	{
		value: 500,
		label: '500',
		text: '500 is past the E4M3 range, so the non-saturating conversion gives NaN. FP8 training scales each tensor so its largest value lands near 448, then converts with saturation (__NV_SATFINITE in CUDA), which clamps anything left over to 448.',
	},
	{
		value: 57344,
		label: '57344',
		text: '57344 is the largest E5M2 number. E5M2 has FP16\'s 5 exponent bits and only 2 mantissa bits: the range of FP16 at a quarter of the precision steps. FP8 training uses E5M2 for gradients and E4M3 for weights and activations.',
	},
	{
		value: 1e-5,
		label: '1e-5',
		text: '1e-5 is below the smallest normal FP16 number, 2^-14 = 6.1e-5, so FP16 stores it as a subnormal with fewer significant bits. A small gradient like this is why FP16 training uses loss scaling.',
	},
	{
		value: 2 ** -25,
		label: '2^-25',
		text: '2^-25 is exactly half the smallest FP16 subnormal, 2^-24. The tie goes to the even neighbour, 0, so FP16 underflows to zero. BF16 and TF32 share FP32\'s exponent range and keep it.',
	},
	{
		value: 3e38,
		label: '3e38',
		text: '3e38 is close to the FP32 maximum of 3.4e38. Only FP32, TF32 and BF16 hold it. TF32 is FP32\'s range with FP16\'s 10 mantissa bits.',
	},
];

export const LAST_STEP = TOUR.length;

export interface Config {
	value: number;
	edit: FormatId;
}

export const convertAll = (x: number) => FORMATS.map((f) => convert(f, x));

export function describe(c: Config, step: number) {
	const stop = TOUR[step - 1];
	if (step === 0) return 'Step 0: type a number, drag the magnitude slider, or click a bit. Step forward for a tour of the edge cases.';
	if (stop && Object.is(stop.value, c.value)) return `Step ${step}, x = ${stop.label}: ${stop.text}`;
	const rows = convertAll(c.value).map((r, i) => `${FORMATS[i].name} ${fmtValue(r.value)} (${kindText(r)})`);
	return `x = ${fmtValue(c.value)}: ${rows.join('; ')}.`;
}

/** Parse the reader's text: decimals, exponents, inf, nan, and 2^k. */
export function parseValue(s: string): number | null {
	const t = s.trim().toLowerCase().replace(/\s+/g, '');
	if (t === '') return null;
	if (/^[+-]?(inf|infinity)$/.test(t)) return t.startsWith('-') ? -Infinity : Infinity;
	if (t === 'nan') return NaN;
	const pow = /^([+-]?)2\^([+-]?\d+)$/.exec(t);
	if (pow) return (pow[1] === '-' ? -1 : 1) * 2 ** Number(pow[2]);
	if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(t)) return null;
	return Number(t);
}

/** Range of the magnitude slider, log10 |x|. */
export const LOG10_MIN = -12;
export const LOG10_MAX = 39;
export const LOG10_STEP = 0.05;

/** log2 axis of the range chart. */
export const AXIS_MIN = -150;
export const AXIS_MAX = 130;
