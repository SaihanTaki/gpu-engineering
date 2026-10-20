/**
 * One block of a matrix multiply C = A x B, n x n, with T x T threads per
 * block: thread (ty, tx) computes C[row][col]. Without shared memory every
 * thread reads its whole row of A and column of B from global memory. With
 * shared memory the block walks the n / T phases of the tiled kernel in the
 * CUDA Programming Guide's shared memory example: every thread loads one
 * element of an A tile and one of a B tile, the block syncs, then all threads
 * read the two tiles T times each from shared memory.
 *
 * Reads are counted as the element loads the kernel issues. Caches (chapter 9)
 * can serve some repeated global loads; the model leaves them out.
 */
export type Mode = 'global' | 'shared';
export const MODES: Mode[] = ['global', 'shared'];
export const SIZES = [8, 16];
export const TILES = [2, 4, 8];

export interface Config {
	mode: Mode;
	n: number;
	tile: number;
}

export const tilesPerSide = (c: Config) => c.n / c.tile;
/** The block drawn: the top-right output tile, so its A strip and B strip sit apart. */
export const blockOf = (c: Config) => ({ bx: tilesPerSide(c) - 1, by: 0 });

export type Kind = 'start' | 'load' | 'sync1' | 'compute' | 'sync2' | 'store';
export interface Step {
	kind: Kind;
	/** Tiled phase, or -1. */
	phase: number;
	/** Inner-loop index k: within the tile when shared, over all of n when global. */
	k: number;
	/** 1-based code lines to highlight. */
	lines: number[];
}

export const CODE: Record<Mode, string> = {
	global: `__global__ void matmul(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}`,
	shared: `__global__ void matmul(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int p = 0; p < n / TILE; ++p) {
        As[ty][tx] = A[row * n + p * TILE + tx];
        Bs[ty][tx] = B[(p * TILE + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * n + col] = sum;
}`,
};

export function steps(c: Config): Step[] {
	const out: Step[] = [{ kind: 'start', phase: -1, k: -1, lines: [] }];
	if (c.mode === 'global') {
		for (let k = 0; k < c.n; k++) out.push({ kind: 'compute', phase: -1, k, lines: [5, 6] });
		out.push({ kind: 'store', phase: -1, k: -1, lines: [7] });
		return out;
	}
	for (let p = 0; p < tilesPerSide(c); p++) {
		out.push({ kind: 'load', phase: p, k: -1, lines: [9, 10] });
		out.push({ kind: 'sync1', phase: p, k: -1, lines: [11] });
		for (let k = 0; k < c.tile; k++) out.push({ kind: 'compute', phase: p, k, lines: [12, 13] });
		out.push({ kind: 'sync2', phase: p, k: -1, lines: [14] });
	}
	out.push({ kind: 'store', phase: -1, k: -1, lines: [16] });
	return out;
}

export const lastStep = (c: Config) => steps(c).length - 1;

/** Element loads issued by one block over the whole kernel. */
export function blockTotals(c: Config) {
	const threads = c.tile * c.tile;
	return {
		threads,
		globalWithout: threads * 2 * c.n,
		globalWith: tilesPerSide(c) * 2 * threads,
		shared: threads * 2 * c.n,
	};
}

/** Element loads issued by the whole grid: (n / T)^2 blocks. */
export function gridTotals(c: Config) {
	const blocks = tilesPerSide(c) ** 2;
	const b = blockTotals(c);
	return { blocks, globalWithout: blocks * b.globalWithout, globalWith: blocks * b.globalWith };
}

/** Bytes of shared memory the tiled kernel declares per block, for float data. */
export const sharedBytes = (c: Config) => 2 * c.tile * c.tile * 4;

export interface State {
	step: number;
	cur: Step;
	globalReads: number;
	sharedReads: number;
	/** Global loads of each A and B element by this block so far, indexed [row * n + col]. */
	aCount: number[];
	bCount: number[];
	/** Shared tile contents: the global (row, col) each As / Bs slot holds, or null before the first load. */
	tileA: { r0: number; c0: number } | null;
	tileB: { r0: number; c0: number } | null;
	/** Reads of each shared slot in the current phase, indexed [r * T + c]. */
	asReads: number[];
	bsReads: number[];
	/** Terms of the dot product each thread has added so far. */
	terms: number;
	stored: boolean;
}

export function stateAt(c: Config, step: number): State {
	const all = steps(c);
	const s = Math.max(0, Math.min(step, all.length - 1));
	const { bx, by } = blockOf(c);
	const T = c.tile;
	const n = c.n;
	const st: State = {
		step: s,
		cur: all[s],
		globalReads: 0,
		sharedReads: 0,
		aCount: new Array(n * n).fill(0),
		bCount: new Array(n * n).fill(0),
		tileA: null,
		tileB: null,
		asReads: new Array(T * T).fill(0),
		bsReads: new Array(T * T).fill(0),
		terms: 0,
		stored: false,
	};
	for (let i = 1; i <= s; i++) {
		const x = all[i];
		if (x.kind === 'compute' && c.mode === 'global') {
			// Each of the T threads in a row loads A[row][k]; each of the T in a column loads B[k][col].
			for (let t = 0; t < T; t++) {
				st.aCount[(by * T + t) * n + x.k] += T;
				st.bCount[x.k * n + bx * T + t] += T;
			}
			st.globalReads += 2 * T * T;
			st.terms++;
		} else if (x.kind === 'load') {
			st.tileA = { r0: by * T, c0: x.phase * T };
			st.tileB = { r0: x.phase * T, c0: bx * T };
			for (let r = 0; r < T; r++)
				for (let q = 0; q < T; q++) {
					st.aCount[(by * T + r) * n + x.phase * T + q]++;
					st.bCount[(x.phase * T + r) * n + bx * T + q]++;
				}
			st.globalReads += 2 * T * T;
			st.asReads.fill(0);
			st.bsReads.fill(0);
		} else if (x.kind === 'compute') {
			// Thread (ty, tx) reads As[ty][k] and Bs[k][tx]; T threads share each.
			for (let t = 0; t < T; t++) {
				st.asReads[t * T + x.k] += T;
				st.bsReads[x.k * T + t] += T;
			}
			st.sharedReads += 2 * T * T;
			st.terms++;
		} else if (x.kind === 'store') {
			st.stored = true;
		}
	}
	return st;
}

const fmt = (x: number) => x.toLocaleString('en-US');
const plural = (x: number, word: string) => `${fmt(x)} ${word}${x === 1 ? '' : 's'}`;

export function describe(c: Config, step: number): string {
	const st = stateAt(c, step);
	const T = c.tile;
	const b = blockTotals(c);
	const { bx, by } = blockOf(c);
	const x = st.cur;
	const blockName = `Block (${bx}, ${by})`;
	switch (x.kind) {
		case 'start':
			return c.mode === 'global'
				? `${blockName} has ${b.threads} threads, one per element of its ${T} x ${T} tile of C. Each thread needs a row of A and a column of B, ${c.n} elements each. Predict: how many global loads does the block issue in total? Then press Next.`
				: `${blockName} has ${b.threads} threads and two ${T} x ${T} tiles in shared memory, As and Bs (${fmt(sharedBytes(c))} bytes). It will walk ${plural(tilesPerSide(c), 'phase')}. Predict: how many global loads does the block issue now? Then press Next.`;
		case 'compute':
			if (c.mode === 'global')
				return `k = ${x.k}: every thread loads A[row][${x.k}] and B[${x.k}][col] from global memory, ${2 * b.threads} loads. The ${T} threads of a row ask for the same A element, and the ${T} threads of a column for the same B element. Global loads so far: ${fmt(st.globalReads)}.`;
			return `Phase ${x.phase}, k = ${x.k}: every thread reads As[ty][${x.k}] and Bs[${x.k}][tx] from shared memory. Each of those ${2 * T} values is read by ${T} threads. No global loads. Shared reads so far: ${fmt(st.sharedReads)}.`;
		case 'load':
			return `Phase ${x.phase}: every thread loads one element of A into As and one of B into Bs, ${2 * b.threads} global loads for the whole block. Each element of the two tiles is loaded once. Global loads so far: ${fmt(st.globalReads)}.`;
		case 'sync1':
			return `Phase ${x.phase}: __syncthreads(). No thread reads the tiles until every thread of the block has written its element, so no one reads a slot that is still empty.`;
		case 'sync2':
			return x.phase === tilesPerSide(c) - 1
				? `Phase ${x.phase}: __syncthreads(). This was the last phase, so no tile will be overwritten; the barrier is kept because the loop body is the same every time.`
				: `Phase ${x.phase}: __syncthreads(). No thread loads the next tile until every thread has finished reading this one, so no one overwrites a value another thread still needs.`;
		case 'store': {
			const ratio = b.globalWithout / b.globalWith;
			return c.mode === 'global'
				? `Every thread writes its sum to C. Done: ${fmt(st.globalReads)} global loads for ${b.threads} outputs, ${2 * c.n} per output. Switch to shared memory to compare.`
				: `Every thread writes its sum to C. Done: ${fmt(st.globalReads)} global loads instead of ${fmt(b.globalWithout)}, ${ratio} times fewer, plus ${fmt(st.sharedReads)} shared memory reads.`;
		}
	}
}
