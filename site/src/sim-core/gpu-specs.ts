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
		sfuPerSm: 16,
		registerFileKbPerSm: 256,
		l1SharedKbPerSm: 192,
		maxSharedKbPerSm: 164,
		l2Mb: 40,
		memory: 'HBM2e',
		memoryGb: 80,
		memoryGbPerS: 2039,
		maxWarpsPerSm: 64,
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
		sources: [
			'https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/',
			'https://www.nvidia.com/en-us/data-center/h100/',
			'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
			'https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html',
		],
	},
];

export const gpuById = (id: string) => GPUS.find((g) => g.id === id) ?? GPUS[0];
