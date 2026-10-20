/**
 * Batcher's bitonic sorting network for n = 2^p keys, one stage per step.
 * Stage (k, j), for k = 2, 4, ..., n and j = k/2, ..., 1, pairs key i with
 * key i + j for every i whose bit j is clear, and orders the pair ascending
 * when bit k of i is clear, descending otherwise. The network has
 * p(p+1)/2 stages of n/2 comparators each.
 * Source: K. E. Batcher, "Sorting networks and their applications",
 * AFIPS Spring Joint Computer Conference 1968, pp. 307-314, doi:10.1145/1468075.1468121.
 */
export type Size = 8 | 16;
export const SIZES: Size[] = [8, 16];

export type InputId = 'random' | 'sorted' | 'reversed';
export const INPUTS: Record<InputId, string> = {
	random: 'Random',
	sorted: 'Already sorted',
	reversed: 'Reversed',
};
export const INPUT_IDS = Object.keys(INPUTS) as InputId[];

export interface Config {
	n: Size;
	input: InputId;
	/** Seed for the random input, so a reload shows the same keys. */
	seed: number;
}

export interface Comparator {
	/** Lower wire; gets the smaller key when `up`. */
	i: number;
	/** Upper wire, i + j. */
	p: number;
	up: boolean;
}

export interface Stage {
	k: number;
	j: number;
	comparators: Comparator[];
}

export const log2 = (n: number) => Math.round(Math.log2(n));

/** Lower key of comparator t in a stage of distance j, as in the kernels. */
export const lowerKey = (t: number, j: number) => Math.floor(t / j) * 2 * j + (t % j);

export function stages(n: number): Stage[] {
	const out: Stage[] = [];
	for (let k = 2; k <= n; k *= 2) {
		for (let j = k >> 1; j > 0; j >>= 1) {
			const comparators = Array.from({ length: n / 2 }, (_, t) => {
				const i = lowerKey(t, j);
				return { i, p: i + j, up: (i & k) === 0 };
			});
			out.push({ k, j, comparators });
		}
	}
	return out;
}

export const stageCount = (n: number) => (log2(n) * (log2(n) + 1)) / 2;
export const comparatorCount = (n: number) => (n / 2) * stageCount(n);

/** Small deterministic generator (mulberry32), so tests and reloads agree. */
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

/** The keys 1..n in the chosen order; random is a shuffle of them. */
export function inputKeys(c: Config): number[] {
	const keys = Array.from({ length: c.n }, (_, i) => i + 1);
	if (c.input === 'reversed') return keys.reverse();
	if (c.input === 'random') {
		const next = rng(c.seed);
		for (let i = keys.length - 1; i > 0; i--) {
			const r = Math.floor(next() * (i + 1));
			[keys[i], keys[r]] = [keys[r], keys[i]];
		}
	}
	return keys;
}

/** Applies one stage; returns the new keys and which comparators swapped. */
export function applyStage(keys: number[], st: Stage) {
	const out = keys.slice();
	const swapped = st.comparators.map(({ i, p, up }) => {
		if (out[i] > out[p] === up) {
			[out[i], out[p]] = [out[p], out[i]];
			return true;
		}
		return false;
	});
	return { keys: out, swapped };
}

export const lastStep = (c: Config) => stageCount(c.n);

export interface State {
	step: number;
	/** Keys after `step` stages. */
	keys: number[];
	/** The stage just applied, or null at step 0. */
	stage: Stage | null;
	swapped: boolean[];
	comparatorsSoFar: number;
	swapsSoFar: number;
}

export function stateAt(c: Config, step: number): State {
	const all = stages(c.n);
	const s = Math.max(0, Math.min(step, all.length));
	let keys = inputKeys(c);
	let swapped: boolean[] = [];
	let swapsSoFar = 0;
	for (let x = 0; x < s; x++) {
		const r = applyStage(keys, all[x]);
		keys = r.keys;
		swapped = r.swapped;
		swapsSoFar += swapped.filter(Boolean).length;
	}
	return { step: s, keys, stage: s ? all[s - 1] : null, swapped, comparatorsSoFar: (s * c.n) / 2, swapsSoFar };
}

export const isSorted = (keys: number[]) => keys.every((v, i) => i === 0 || keys[i - 1] <= v);

/** Code lines shown next to the sim: the host loop and the kernel it launches. */
export const CODE = `for (int k = 2; k <= n; k *= 2)
    for (int j = k / 2; j > 0; j /= 2)
        bitonic_step<<<blocks, THREADS>>>(a, n, j, k);

__global__ void bitonic_step(float *a, int n, int j, int k) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n / 2) return;
    int i = (t / j) * 2 * j + t % j;
    compare_swap(a[i], a[i + j], (i & k) == 0);
}`;

/** Lines lit for a stage: the launch, and the kernel's compare-swap. */
export const ACTIVE_LINES = [3, 9];

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	const S = stageCount(c.n);
	if (!st.stage) {
		return `${c.n} keys on ${c.n} wires, ${INPUTS[c.input].toLowerCase()}. Predict: how many stages and how many comparators does the network need, and does the answer change for sorted input? Then press Next.`;
	}
	const { k, j } = st.stage;
	const swaps = st.swapped.filter(Boolean).length;
	const phase = log2(k);
	let text = `Stage ${st.step} of ${S}, k = ${k}, j = ${j}: ${c.n / 2} comparators each join wire i to wire i + ${j}. `;
	text +=
		k === c.n
			? 'Every pair is ordered ascending. '
			: `Groups of ${k} wires alternate: where i & ${k} is 0 the pair orders ascending, otherwise descending. `;
	text += `${swaps} of ${c.n / 2} swapped. Comparators so far: ${st.comparatorsSoFar} of ${comparatorCount(c.n)}.`;
	if (j === 1) {
		text +=
			k === c.n
				? ` Done: ${S} stages, ${comparatorCount(c.n)} comparators, ${st.swapsSoFar} swaps, and the keys are sorted.`
				: ` Phase ${phase} ends: each run of ${k} keys is sorted, alternating up and down, so each run of ${2 * k} is bitonic.`;
	}
	return text;
}

/**
 * Kernel launches to sort m keys (m a power of two, m >= tile) the two ways the
 * chapter's kernels do it: one launch per stage, or shared memory for every
 * stage with j < tile and a global launch only for j >= tile.
 */
export function launchCounts(m: number, tile: number) {
	let hybridGlobal = 0;
	let hybridShared = 1;
	for (let k = 2 * tile; k <= m; k *= 2) {
		for (let j = k >> 1; j >= tile; j >>= 1) hybridGlobal++;
		hybridShared++;
	}
	return { perStage: stageCount(m), hybridGlobal, hybridShared };
}
