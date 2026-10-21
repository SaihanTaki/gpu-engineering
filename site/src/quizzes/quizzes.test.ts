import { describe, expect, it } from 'vitest';
import type { Question } from './types';

const files = import.meta.glob<{ default: Question[] }>('./*.ts', { eager: true });
const quizzes = Object.entries(files).filter(([path]) => !/types|test/.test(path));

it('collects quiz files', () => expect(Array.isArray(quizzes)).toBe(true));

describe.each(quizzes)('%s', (_path, mod) => {
	const qs = mod.default;
	it('has 10 to 30 questions', () => {
		expect(qs.length).toBeGreaterThanOrEqual(10);
		expect(qs.length).toBeLessThanOrEqual(30);
	});
	it.each(qs.map((q, i) => [i, q] as const))('question %i is well formed', (_i, q) => {
		expect(q.q.trim()).not.toBe('');
		expect(q.why.trim()).not.toBe('');
		expect(q.options).toHaveLength(4);
		expect(new Set(q.options.map((o) => o.trim())).size).toBe(4);
		expect([0, 1, 2, 3]).toContain(q.answer);
		expect(`${q.q} ${q.options.join(' ')} ${q.why}`).not.toMatch(/[—–]/);
	});
	it('spreads correct answers over all four positions', () => {
		expect(new Set(qs.map((q) => q.answer)).size).toBe(4);
	});
	// "Pick the longest option" must not beat guessing by much (chance is 25%).
	it('does not give the answer away by length', () => {
		const longest = qs.filter((q) => {
			const others = q.options.filter((_, i) => i !== q.answer).map((o) => o.length);
			return q.options[q.answer].length > Math.max(...others);
		});
		expect(longest.length, `correct option is longest in:\n${longest.map((q) => q.q).join('\n')}`).toBeLessThanOrEqual(
			Math.ceil(qs.length * 0.3),
		);
	});
});
