// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Set these for your fork or deployment. GitHub Pages project sites need
// SITE_BASE=/<repo-name>; a custom domain or local dev uses '/'.
const REPO_URL = process.env.REPO_URL ?? 'https://github.com/OWNER/gpu-engineering';
const SITE_URL = process.env.SITE_URL ?? 'https://owner.github.io';
const SITE_BASE = process.env.SITE_BASE ?? '/';

const part = (label, directory) => ({
	label,
	collapsed: false,
	autogenerate: { directory },
});

export default defineConfig({
	site: SITE_URL,
	base: SITE_BASE,
	integrations: [
		starlight({
			title: 'GPU Engineering',
			description:
				'Learn GPU architecture and CUDA programming through interactive simulations, with every kernel in CUDA C++, Numba and Triton.',
			logo: { src: './src/assets/logo.svg', replacesTitle: false },
			social: [{ icon: 'github', label: 'GitHub', href: REPO_URL }],
			editLink: { baseUrl: `${REPO_URL}/edit/main/site/` },
			lastUpdated: true,
			customCss: ['./src/styles/custom.css'],
			expressiveCode: {
				themes: ['github-dark', 'github-light'],
				styleOverrides: { borderRadius: '0.5rem' },
			},
			sidebar: [
				{ label: 'Start here', items: [{ label: 'How this course works', slug: 'start' }] },
				part('Part 0 · Why GPUs?', 'part-0-why-gpus'),
				part('Part 1 · Foundations', 'part-1-foundations'),
				part('Part 2 · Memory hierarchy', 'part-2-memory'),
				part('Part 3 · Parallel patterns', 'part-3-patterns'),
				part('Part 4 · Performance engineering', 'part-4-performance'),
				part('Part 5 · AI kernels', 'part-5-ai-kernels'),
				{ ...part('Appendices', 'appendix'), collapsed: true },
			],
		}),
	],
	vite: {
		// Chapters import kernel source from ../kernels with ?raw.
		server: { fs: { allow: ['..'] } },
	},
});
