import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	resolve: { alias: { '@sim-core': fileURLToPath(new URL('./src/sim-core', import.meta.url)) } },
});
