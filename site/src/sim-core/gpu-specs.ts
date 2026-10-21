/**
 * Hardware numbers used by the simulations. Each preset is the shipped
 * product, not the full die. Sources are listed per preset.
 */
export interface GpuSpec {
	id: 't4' | 'a100' | 'h100';
	name: string;
	architecture: string;
	computeCapability: string;
	sms: number;
	partitionsPerSm: number;
	fp32PerSm: number;
	int32PerSm: number;
	fp64PerSm: number;
	tensorPerSm: number;
	tensorGeneration: string;
	sfuPerSm: number;
	registerFileKbPerSm: number;
	l1SharedKbPerSm: number;
	maxSharedKbPerSm: number;
	l2Mb: number;
	memory: string;
	memoryGb: number;
	memoryGbPerS: number;
	maxWarpsPerSm: number;
	/** Occupancy limits; see OCCUPANCY_SOURCES. */
	maxBlocksPerSm: number;
	maxThreadsPerBlock: number;
	maxRegsPerThread: number;
	maxRegsPerBlock: number;
	/** Registers are given to a warp in multiples of this many. */
	regAllocUnit: number;
	/** Bytes; a block's shared memory is rounded up to a multiple of this. */
	smemAllocUnit: number;
	/** Bytes of shared memory the driver takes for every resident block. */
	reservedSmemPerBlock: number;
	/** Largest shared memory one block can opt in to, KB, not counting the reserved part. */
	maxSharedKbPerBlock: number;
	pcieGen: 3 | 4 | 5;
	/** Peak PCIe x16 bandwidth in one direction, GB/s, rounded (real: 15.75, 31.5, 63). */
	pcieGbPerS: number;
	/** Peak rates at boost clock, dense (no structured sparsity). TFLOPS, or TOPS for INT8. */
	peakFp32Tflops: number;
	peakFp64Tflops: number;
	/** Null where the tensor cores have no TF32 mode. */
	peakTf32TensorTflops: number | null;
	/** FP16 inputs with FP32 accumulate; BF16 runs at the same rate where supported. */
	peakFp16TensorTflops: number;
	/** Null where the tensor cores have no FP8 mode. */
	peakFp8TensorTflops: number | null;
	peakInt8TensorTops: number;
	sources: string[];
}

export const GPUS: GpuSpec[] = [
	{
		id: 't4',
		name: 'Tesla T4',
		architecture: 'Turing',
		computeCapability: '7.5',
		sms: 40,
		partitionsPerSm: 4,
		fp32PerSm: 64,
		int32PerSm: 64,
		fp64PerSm: 2,
		tensorPerSm: 8,
		tensorGeneration: 'second',
		// From the guide's throughput table: 16 results per clock per SM for SFU operations.
		sfuPerSm: 16,
		registerFileKbPerSm: 256,
		l1SharedKbPerSm: 96,
		maxSharedKbPerSm: 64,
		// NVIDIA states 4 MB for the TU104 chip; no separate T4 figure is published.
		l2Mb: 4,
		memory: 'GDDR6',
		memoryGb: 16,
		// Product datasheet figure; the Turing whitepaper lists 320 GB/s.
		memoryGbPerS: 300,
		maxWarpsPerSm: 32,
		// Turing (7.5): 16 resident blocks, 256-byte shared memory granularity, nothing reserved per block.
		maxBlocksPerSm: 16,
		maxThreadsPerBlock: 1024,
		maxRegsPerThread: 255,
		maxRegsPerBlock: 65536,
		regAllocUnit: 256,
		smemAllocUnit: 256,
		reservedSmemPerBlock: 0,
		maxSharedKbPerBlock: 64,
		pcieGen: 3,
		// Datasheet: x16 PCIe Gen3, 32 GB/s. One direction of Gen3 x16 is 8 GT/s x 16 x 128/130 / 8 = 15.75 GB/s, so 32 sums both; we use half.
		pcieGbPerS: 16,
		// Datasheet: single precision 8.1 TFLOPS.
		peakFp32Tflops: 8.1,
		// Not published for T4. Turing whitepaper: FP64 runs at 1/32 of the FP32 rate; 40 SMs x 2 FP64 units x 2 x 1.59 GHz = 0.254.
		peakFp64Tflops: 0.25,
		// Turing tensor cores have no TF32 mode (introduced with Ampere).
		peakTf32TensorTflops: null,
		// Turing whitepaper Table 5: 65 TFLOPS with FP16 or FP32 accumulate; datasheet "Mixed-Precision (FP16/FP32) 65 TFLOPS". No BF16 on Turing.
		peakFp16TensorTflops: 65,
		peakFp8TensorTflops: null,
		// Datasheet: INT8 130 TOPS.
		peakInt8TensorTops: 130,
		sources: [
			'https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf',
			'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf',
			'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
			'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html',
		],
	},
	{
		id: 'a100',
		name: 'A100 SXM4 80 GB',
		architecture: 'Ampere',
		computeCapability: '8.0',
		sms: 108,
		partitionsPerSm: 4,
		fp32PerSm: 64,
		int32PerSm: 64,
		fp64PerSm: 32,
		tensorPerSm: 4,
		tensorGeneration: 'third',
		// From the guide's throughput table: 16 results per clock per SM for SFU operations.
		sfuPerSm: 16,
		registerFileKbPerSm: 256,
		l1SharedKbPerSm: 192,
		maxSharedKbPerSm: 164,
		l2Mb: 40,
		memory: 'HBM2e',
		memoryGb: 80,
		memoryGbPerS: 2039,
		maxWarpsPerSm: 64,
		// Ampere (8.0): 32 resident blocks, 128-byte granularity, 1 KB reserved per block, so 164 KB per SM but 163 KB per block.
		maxBlocksPerSm: 32,
		maxThreadsPerBlock: 1024,
		maxRegsPerThread: 255,
		maxRegsPerBlock: 65536,
		regAllocUnit: 256,
		smemAllocUnit: 128,
		reservedSmemPerBlock: 1024,
		maxSharedKbPerBlock: 163,
		pcieGen: 4,
		// Datasheet: PCIe Gen4 64 GB/s, both directions summed (Gen4 x16 is 31.5 GB/s each way); we use half.
		pcieGbPerS: 32,
		// Datasheet (Jun 2021): FP32 19.5, FP64 9.7 TFLOPS.
		peakFp32Tflops: 19.5,
		peakFp64Tflops: 9.7,
		// Datasheet dense figures; the starred values twice as large are with sparsity.
		peakTf32TensorTflops: 156,
		// Datasheet: FP16 and BF16 tensor 312 TFLOPS; Ampere whitepaper lists 312 with FP32 accumulate too.
		peakFp16TensorTflops: 312,
		// FP8 tensor cores arrived with Hopper.
		peakFp8TensorTflops: null,
		peakInt8TensorTops: 624,
		sources: [
			'https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf',
			'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf',
			'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
			'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html',
		],
	},
	{
		id: 'h100',
		name: 'H100 SXM5 80 GB',
		architecture: 'Hopper',
		computeCapability: '9.0',
		sms: 132,
		partitionsPerSm: 4,
		fp32PerSm: 128,
		int32PerSm: 64,
		fp64PerSm: 64,
		tensorPerSm: 4,
		tensorGeneration: 'fourth',
		// From the guide's throughput table: 16 results per clock per SM for SFU operations.
		sfuPerSm: 16,
		registerFileKbPerSm: 256,
		l1SharedKbPerSm: 256,
		maxSharedKbPerSm: 228,
		l2Mb: 50,
		memory: 'HBM3',
		memoryGb: 80,
		// Product page figure; the pre-launch Hopper whitepaper lists 3000 GB/s.
		memoryGbPerS: 3350,
		maxWarpsPerSm: 64,
		// Hopper (9.0): 32 resident blocks, 128-byte granularity, 1 KB reserved per block, so 228 KB per SM but 227 KB per block.
		maxBlocksPerSm: 32,
		maxThreadsPerBlock: 1024,
		maxRegsPerThread: 255,
		maxRegsPerBlock: 65536,
		regAllocUnit: 256,
		smemAllocUnit: 128,
		reservedSmemPerBlock: 1024,
		maxSharedKbPerBlock: 227,
		pcieGen: 5,
		// Product page: PCIe Gen5 128 GB/s, both directions summed (Gen5 x16 is 63 GB/s each way); we use half.
		pcieGbPerS: 64,
		// Product page, H100 SXM: FP32 67, FP64 34 TFLOPS. The pre-launch whitepaper lists 60 and 30 at an unfinalized clock.
		peakFp32Tflops: 67,
		peakFp64Tflops: 34,
		// Product page tensor figures are with sparsity (TF32 989, FP16/BF16 1,979, FP8 3,958, INT8 3,958 TOPS); dense is half.
		peakTf32TensorTflops: 494.5,
		// Hopper whitepaper: FP16 tensor rate is the same with FP16 or FP32 accumulate.
		peakFp16TensorTflops: 989.5,
		peakFp8TensorTflops: 1979,
		peakInt8TensorTops: 1979,
		sources: [
			'https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/',
			'https://www.nvidia.com/en-us/data-center/h100/',
			'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
			'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html',
		],
	},
];

/**
 * Sources for the occupancy fields. The compute capability tables give the
 * per-SM and per-block limits. Allocation units and the per-block reservation
 * are what cuda_occupancy.h, the CUDA Toolkit header behind the runtime's
 * occupancy API, uses (cudaOccRegAllocationGranularity,
 * cudaOccSMemAllocationGranularity, reservedSharedMemPerBlock); the 12.4
 * programming guide states the 1 KB reservation for 8.x and 9.0.
 */
export const OCCUPANCY_SOURCES = [
	'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
	'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#shared-memory-8-x',
	'https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html',
];

export const gpuById = (id: string) => GPUS.find((g) => g.id === id) ?? GPUS[0];
