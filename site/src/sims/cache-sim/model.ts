/**
 * One thread's stream of 4-byte loads going through a small L1 and L2.
 *
 * Scaled down on purpose: the caches hold a handful of lines so every line
 * fits on screen. Both levels are modeled as fully associative with LRU
 * replacement. That is an assumption: NVIDIA does not document the
 * associativity, replacement policy or address hashing of its L1 and L2.
 */
export const ELEM_BYTES = 4;
/** One 32-byte sector. Real L1 and L2 lines are 128 bytes made of four sectors filled separately. */
export const LINE_BYTES = 32;
export const PER_LINE = LINE_BYTES / ELEM_BYTES;
/** CUDA Programming Guide, compute capability tables: constant memory size and constant cache working set per SM. */
export const CONSTANT_MEMORY_BYTES = 64 * 1024;
export const CONSTANT_CACHE_BYTES = 8 * 1024;

export const PATTERNS = ['sequential', 'strided', 'window', 'random'] as const;
export type Pattern = (typeof PATTERNS)[number];

export interface Config {
	pattern: Pattern;
	/** Stride (strided), window size (window) or range (random), in elements. Unused for sequential. */
	param: number;
	accesses: number;
	l1Lines: number;
	l2Lines: number;
}

export type Outcome = 'l1' | 'l2' | 'dram';

export interface Access {
	element: number;
	line: number;
	outcome: Outcome;
	/** Cache contents after this access, most recently used first. */
	l1: number[];
	l2: number[];
	l1Evicted: number | null;
	l2Evicted: number | null;
}

export const PARAM: Record<Pattern, { label: string; min: number; max: number; step: number } | null> = {
	sequential: null,
	strided: { label: 'Stride (elements)', min: 1, max: 32, step: 1 },
	window: { label: 'Window (elements)', min: 8, max: 256, step: 8 },
	random: { label: 'Range (elements)', min: 8, max: 1024, step: 8 },
};

/** Deterministic PRNG (mulberry32), so a random stream is the same on every load and in the tests. */
function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Element index read by each access i. */
export function stream(c: Config): number[] {
	const next = rng(9);
	return Array.from({ length: c.accesses }, (_, i) => {
		switch (c.pattern) {
			case 'sequential':
				return i;
			case 'strided':
				return i * c.param;
			case 'window':
				return i % c.param;
			case 'random':
				return Math.floor(next() * c.param);
		}
	});
}

/** Look up a line in an LRU list (most recent first). Returns whether it hit and what was evicted. */
function touch(cache: number[], line: number, capacity: number) {
	const at = cache.indexOf(line);
	if (at >= 0) {
		cache.splice(at, 1);
		cache.unshift(line);
		return { hit: true, evicted: null };
	}
	cache.unshift(line);
	return { hit: false, evicted: cache.length > capacity ? cache.pop()! : null };
}

export function simulate(c: Config): Access[] {
	const l1: number[] = [];
	const l2: number[] = [];
	return stream(c).map((element) => {
		const line = Math.floor(element / PER_LINE);
		let outcome: Outcome = 'l1';
		let l2Evicted: number | null = null;
		const a = touch(l1, line, c.l1Lines);
		// Only L1 misses reach L2; a miss there fetches the line from DRAM into both levels.
		if (!a.hit) {
			const b = touch(l2, line, c.l2Lines);
			outcome = b.hit ? 'l2' : 'dram';
			l2Evicted = b.evicted;
		}
		return { element, line, outcome, l1: [...l1], l2: [...l2], l1Evicted: a.evicted, l2Evicted };
	});
}

export interface Totals {
	accesses: number;
	l1: number;
	l2: number;
	dram: number;
	dramBytes: number;
}

/** Counts over the first `step` accesses. */
export function totals(sim: Access[], step: number): Totals {
	const t: Totals = { accesses: step, l1: 0, l2: 0, dram: 0, dramBytes: 0 };
	for (const a of sim.slice(0, step)) t[a.outcome]++;
	t.dramBytes = t.dram * LINE_BYTES;
	return t;
}

export const pct = (part: number, whole: number) => (whole ? `${+((100 * part) / whole).toFixed(1)}%` : 'n/a');
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function describe(c: Config, sim: Access[], step: number): string {
	if (step === 0) {
		return `Nothing read yet. L1 holds ${plural(c.l1Lines, 'line')} and L2 ${plural(c.l2Lines, 'line')} of ${LINE_BYTES} bytes, all empty. Guess the L1 hit rate, then press Play or Next.`;
	}
	const a = sim[step - 1];
	const where = `Access ${step} of ${sim.length}: a[${a.element}], byte ${a.element * ELEM_BYTES}, line ${a.line}.`;
	let what: string;
	if (a.outcome === 'l1') what = 'L1 hit.';
	else if (a.outcome === 'l2') what = `L1 miss, L2 hit: line ${a.line} copied into L1${a.l1Evicted !== null ? `, which evicts line ${a.l1Evicted}` : ''}.`;
	else {
		const ev = [a.l1Evicted !== null ? `L1 evicts line ${a.l1Evicted}` : '', a.l2Evicted !== null ? `L2 evicts line ${a.l2Evicted}` : '']
			.filter(Boolean)
			.join(', ');
		what = `Miss in L1 and L2: line ${a.line} read from DRAM (${LINE_BYTES} bytes) into L2 and L1${ev ? `; ${ev}` : ''}.`;
	}
	const t = totals(sim, step);
	return `${where} ${what} So far: L1 hits ${t.l1}, L2 hits ${t.l2}, DRAM reads ${t.dram}.`;
}
