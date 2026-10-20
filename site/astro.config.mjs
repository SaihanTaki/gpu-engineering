// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Set these for your fork or deployment. GitHub Pages project sites need
// SITE_BASE=/<repo-name>; a custom domain or local dev uses '/'.
const REPO_URL = process.env.REPO_URL ?? 'https://github.com/OWNER/gpu-engineering';
const SITE_URL = process.env.SITE_URL ?? 'https://owner.github.io';
const SITE_BASE = process.env.SITE_BASE ?? '/';

export default defineConfig({
	site: SITE_URL,
	base: SITE_BASE,
	integrations: [
		starlight({
			title: 'GPU Engineering',
			description:
				'Learn GPU architecture and CUDA programming through interactive simulations, with every kernel in CUDA C++, Numba and Triton.',
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
				// Add each part here when its first chapter exists, so no group is empty.
				{ label: 'Part 0: Why GPUs?', items: [{ autogenerate: { directory: 'part-0-why-gpus' } }] },
				{ label: 'Part 1: Foundations', items: [{ autogenerate: { directory: 'part-1-foundations' } }] },
			],
		}),
	],
	vite: {
		// Chapters import kernel source from ../kernels with ?raw.
		server: { fs: { allow: ['..'] } },
	},
});
