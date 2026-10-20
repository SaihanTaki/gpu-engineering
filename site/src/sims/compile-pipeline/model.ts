/**
 * The compilation pipelines of chapter 0.2: one SAXPY kernel, followed from
 * source to machine code along three paths.
 *
 * Listings are files under kernels/ch0-2-software-stack/ written by real tools
 * (see generate.py there). This module only says which file belongs to which
 * stage and which contiguous slice of it to show.
 */

export const PATH_IDS = ['triton', 'numba', 'cuda'] as const;
export type PathId = (typeof PATH_IDS)[number];

/** A contiguous slice: from the first line containing `start` to the next line containing `end`, or `count` lines. */
export interface ExcerptSpec {
	start: string;
	end?: string;
	count?: number;
}

export interface Stage {
	id: string;
	/** Box title in the diagram. Keep it short: it must fit a 100px box. */
	label: string;
	/** Second line of the box: what produced this stage. */
	tool: string;
	/** Listing path relative to kernels/ch0-2-software-stack/. */
	file?: string;
	lang?: string;
	excerpt?: ExcerptSpec;
	/** Shell commands for stages this site cannot generate (no nvcc on the build machine). */
	commands?: string;
	/** Strings in the excerpt the notice talks about. Their lines are highlighted. */
	look?: string[];
	made: string;
	read: string;
	notice: string;
}

export interface Path {
	id: PathId;
	label: string;
	stages: Stage[];
}

const DOCKER = '# Run in the course Docker image, from kernels/ch0-2-software-stack/cuda.\n';

const gpuStage = (made: string, notice: string): Stage => ({
	id: 'gpu',
	label: 'GPU',
	tool: 'driver loads',
	made,
	read: 'The SMs of the GPU run the SASS.',
	notice,
});

export const PATHS: Record<PathId, Path> = {
	triton: {
		id: 'triton',
		label: 'Triton',
		stages: [
			{
				id: 'source',
				label: 'Python',
				tool: 'you write',
				file: 'triton/saxpy.py',
				lang: 'python',
				excerpt: { start: '@triton.jit', end: 'tl.store' },
				look: ['tl.constexpr'],
				made: 'You write the kernel in Python with @triton.jit.',
				read: "Triton's front end walks the Python syntax tree of the function and emits Triton IR.",
				notice:
					'BLOCK is a tl.constexpr, so its value is fixed when the kernel is compiled. These listings use BLOCK = 1024.',
			},
			{
				id: 'ttir',
				label: 'TTIR',
				tool: 'Triton',
				file: 'generated/triton.ttir',
				lang: 'txt',
				excerpt: { start: 'tt.func public @saxpy', end: 'tt.return' },
				look: ['tt.load'],
				made: 'Triton IR (TTIR), an MLIR dialect, made from the Python function.',
				read: 'Triton passes rewrite it into TritonGPU IR.',
				notice:
					'Every value is a tensor<1024x...>, and one tt.load reads the whole block of x. There are no threads in this IR yet.',
			},
			{
				id: 'ttgir',
				label: 'TTGIR',
				tool: 'Triton',
				file: 'generated/triton.ttgir',
				lang: 'txt',
				excerpt: { start: '#blocked', end: 'tt.load' },
				look: ['sizePerThread = [4]'],
				made: 'TritonGPU IR (TTGIR): the same program with a layout that says which thread owns which elements.',
				read: 'Triton lowers it to LLVM IR.',
				notice:
					'The #ttg.blocked layout gives each thread 4 elements, 32 threads per warp and 4 warps: 512 elements in one pass. The block is 1024, so each thread ends up with 8.',
			},
			{
				id: 'llir',
				label: 'LLVM IR',
				tool: 'Triton',
				file: 'generated/triton.llir',
				lang: 'llvm',
				excerpt: { start: 'fmul float', end: 'ret void' },
				look: ['fmul float', 'fadd float'],
				made: "LLVM IR for the NVPTX target, written from one thread's point of view.",
				read: "LLVM's NVPTX back end turns it into PTX.",
				notice:
					'a * x + y is still two instructions per element here: 8 fmul and 8 fadd. The stores are inline PTX (st.global.b32) that Triton wrote into the IR.',
			},
			{
				id: 'ptx',
				label: 'PTX',
				tool: 'LLVM NVPTX',
				file: 'generated/triton.ptx',
				lang: 'txt',
				excerpt: { start: '.version', end: '.reqntid' },
				look: ['.target sm_80', '.reqntid 128'],
				made: 'PTX, the virtual instruction set, written by the LLVM NVPTX back end.',
				read: 'ptxas, which Triton ships, assembles it into a cubin for sm_80.',
				notice:
					'.target sm_80 is the compute capability this PTX was written for. .reqntid 128 fixes the block at 128 threads: 4 warps of 32.',
			},
			{
				id: 'sass',
				label: 'cubin',
				tool: 'ptxas',
				file: 'generated/triton.sass',
				lang: 'txt',
				excerpt: { start: 'FFMA', end: 'STG.E [R4.64+0x80c]' },
				look: ['FFMA', 'STG.E'],
				made: 'A cubin for sm_80, shown as SASS by cuobjdump -sass.',
				read: "Triton's launcher hands the cubin to the driver.",
				notice:
					"LLVM fused each fmul and fadd into one fma.rn.f32 in the PTX, and ptxas turned those into the 8 FFMA here. The stores are STG.E, stores to global memory. The hex comments are each instruction's binary encoding.",
			},
			gpuStage(
				"Triton's launcher loads the cubin with the driver API (cuModuleLoadData) and launches it.",
				'This cubin only runs on compute capability 8.x. On a real launch Triton compiles for the GPU it finds and caches the result in ~/.triton/cache.',
			),
		],
	},
	numba: {
		id: 'numba',
		label: 'Numba',
		stages: [
			{
				id: 'source',
				label: 'Python',
				tool: 'you write',
				file: 'numba/saxpy.py',
				lang: 'python',
				excerpt: { start: '@cuda.jit', end: 'out[i] =' },
				look: ['cuda.grid(1)'],
				made: 'You write the kernel in Python with @cuda.jit.',
				read: 'Numba types the function for its argument types and lowers it to NVVM IR.',
				notice:
					'This is a thread-level kernel, like CUDA C++: cuda.grid(1) is blockIdx.x * blockDim.x + threadIdx.x, and each thread handles one element.',
			},
			{
				id: 'nvvm',
				label: 'NVVM IR',
				tool: 'Numba',
				file: 'generated/numba.ll',
				lang: 'llvm',
				excerpt: { start: 'entry:', end: '"inserted.strides" =' },
				look: ['i64 %"x.2", 2', 'float* %"x.4", 4'],
				made: "NVVM IR, the form of LLVM IR that NVIDIA's NVVM compiler accepts, written by Numba.",
				read: 'NVVM (libNVVM, from the CUDA wheels) optimizes it and writes PTX.',
				notice:
					'Each array argument arrives as seven values that the IR packs into a struct. For x they are x.0 to x.6: x.2 is the element count (nitems) and x.4 is the data pointer.',
			},
			{
				id: 'ptx',
				label: 'PTX',
				tool: 'NVVM',
				file: 'generated/numba.ptx',
				lang: 'txt',
				excerpt: { start: '%tid.x', end: 'ret;' },
				look: ['ld.f32', 'st.f32'],
				made: 'PTX written by NVVM. Its header says .version 9.4 and .target sm_80.',
				read: 'nvJitLink, from the CUDA wheels, compiles it into a cubin.',
				notice:
					'The loads are ld.f32 and the store is st.f32, with no .global. PTX calls this generic addressing: the address goes to global memory unless it falls in the window of shared, local or constant memory.',
			},
			{
				id: 'sass',
				label: 'cubin',
				tool: 'nvJitLink',
				file: 'generated/numba.sass',
				lang: 'txt',
				excerpt: { start: 'LD.E R4', end: 'EXIT ;' },
				look: ['LD.E', 'ST.E', 'FFMA'],
				made: 'A cubin for sm_80, shown as SASS by cuobjdump -sass.',
				read: 'numba-cuda hands the cubin to the driver.',
				notice:
					'LD.E and ST.E are the generic load and store, matching the generic ld.f32 and st.f32 in the PTX. The Triton cubin uses LDG.E and STG.E instead.',
			},
			gpuStage(
				'numba-cuda loads the cubin through the driver and launches it.',
				"generate.py asked for sm_80. On a real GPU, Numba compiles for that GPU's compute capability instead.",
			),
		],
	},
	cuda: {
		id: 'cuda',
		label: 'CUDA C++',
		stages: [
			{
				id: 'source',
				label: 'CUDA C++',
				tool: 'you write',
				file: 'cuda/saxpy.cu',
				lang: 'cpp',
				excerpt: { start: '__global__', count: 6 },
				look: ['__global__'],
				made: 'You write host and device code in one .cu file.',
				read: 'nvcc splits it: host code goes to the host C++ compiler, device code to the GPU compiler.',
				notice: '__global__ marks saxpy as a kernel: device code that the host launches.',
			},
			{
				id: 'ptx',
				label: 'PTX',
				tool: 'nvcc',
				commands: `${DOCKER}make ptx\n# runs: nvcc -O2 -std=c++17 -arch=compute_80 -ptx -o saxpy.ptx saxpy.cu\nhead -20 saxpy.ptx`,
				lang: 'bash',
				made: "nvcc's GPU compiler turns the device code into PTX for the virtual architecture compute_80.",
				read: 'ptxas, called by nvcc, assembles the PTX into a cubin.',
				notice: 'Run the commands, then look for .version and .target sm_80 at the top of saxpy.ptx, as in the Triton and Numba PTX.',
			},
			{
				id: 'sass',
				label: 'cubin',
				tool: 'ptxas',
				commands: `${DOCKER}make sass\n# runs: nvcc -O2 -std=c++17 -arch=compute_80 -code=sm_80 -cubin -o saxpy.cubin saxpy.cu\n#       cuobjdump -sass saxpy.cubin`,
				lang: 'bash',
				made: 'ptxas writes a cubin for the real architecture sm_80. cuobjdump -sass prints its SASS.',
				read: 'nvcc packs cubins and PTX into a fatbin.',
				notice: 'nvcc fuses a multiply and an add into one instruction by default (--fmad=true), so look for one FFMA doing a * x + y.',
			},
			{
				id: 'fatbin',
				label: 'fatbin',
				tool: 'nvcc',
				commands: `${DOCKER}make fatbin\n# runs: nvcc -O2 -std=c++17 -arch=compute_80 -code=sm_80,compute_80 -o saxpy_fat saxpy.cu\n#       cuobjdump -lelf saxpy_fat\n#       cuobjdump -lptx saxpy_fat`,
				lang: 'bash',
				made: 'nvcc embeds a fatbin in the executable. This one holds the sm_80 cubin and the compute_80 PTX.',
				read: 'The CUDA runtime and the driver, when the program runs.',
				notice: 'cuobjdump -lelf lists the cubins in the fatbin and cuobjdump -lptx lists the PTX.',
			},
			gpuStage(
				'The driver picks the best code in the fatbin for the GPU it runs on.',
				'On any 8.x GPU the driver loads the sm_80 cubin. On a newer GPU such as an H100 (9.0) no cubin fits, so the driver JIT compiles the compute_80 PTX. A T4 (7.5) cannot run either.',
			),
		],
	},
};

export const lastStep = (path: PathId) => PATHS[path].stages.length - 1;

export const stageAt = (path: PathId, step: number): Stage => {
	const stages = PATHS[path].stages;
	return stages[Math.min(Math.max(0, Math.round(step)), stages.length - 1)];
};

export interface Excerpt {
	code: string;
	/** 1-based, inclusive line numbers in the whole file. */
	from: number;
	to: number;
	total: number;
}

/** Lines of `text` without the trailing newline's empty last element. */
const linesOf = (text: string) => text.replace(/\n$/, '').split('\n');

export function excerpt(text: string, spec: ExcerptSpec): Excerpt {
	const lines = linesOf(text);
	const start = lines.findIndex((l) => l.includes(spec.start));
	if (start < 0) throw new Error(`excerpt start not found: ${spec.start}`);
	let end: number;
	if (spec.count !== undefined) {
		end = Math.min(lines.length, start + spec.count) - 1;
	} else if (spec.end !== undefined) {
		const rel = lines.slice(start).findIndex((l) => l.includes(spec.end!));
		if (rel < 0) throw new Error(`excerpt end not found: ${spec.end}`);
		end = start + rel;
	} else {
		end = lines.length - 1;
	}
	return { code: lines.slice(start, end + 1).join('\n'), from: start + 1, to: end + 1, total: lines.length };
}

/** 1-based line numbers in `code` that contain any of the strings in `look`. */
export function lookLines(code: string, look: string[] = []): number[] {
	return linesOf(code)
		.map((line, i) => (look.some((s) => line.includes(s)) ? i + 1 : 0))
		.filter((n) => n > 0);
}

export function describe(path: PathId, step: number): string {
	const p = PATHS[path];
	const s = stageAt(path, step);
	const i = p.stages.indexOf(s);
	return `${p.label}, stage ${i + 1} of ${p.stages.length}: ${s.label}. ${s.made}`;
}
