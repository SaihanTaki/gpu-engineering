import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Course fields on top of Starlight's frontmatter. All optional so that
// non-chapter pages (landing page, appendices) validate too.
const course = z.object({
	/** Chapter number as shown in the plan, e.g. "0.1" or "6". */
	chapter: z.string().optional(),
	/** `planned` chapters render as an outline with a "coming soon" banner. */
	status: z.enum(['planned', 'draft', 'ready']).default('planned'),
	/** Chapter numbers the reader should finish first. */
	prerequisites: z.array(z.string()).default([]),
	/** Path under the repo's notebooks/ folder. */
	colab: z.string().optional(),
	/** Path under the repo's kernels/ folder. */
	kernels: z.string().optional(),
});

export const collections = {
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema({ extend: course }) }),
};
