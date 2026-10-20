/**
 * Highlights lines in a code block rendered by SimFrame (Expressive Code).
 * Line numbers are 1-based, matching the source shown on the page.
 */
export function highlightLines(container: Element | null, lines: number[]) {
	if (!container) return;
	const rows = container.querySelectorAll('.ec-line');
	const active = new Set(lines);
	rows.forEach((row, i) => {
		row.classList.toggle('sim-line-active', active.has(i + 1));
	});
}

/** Show only the code variant whose data-variant matches `name`. */
export function showVariant(frame: Element, name: string) {
	frame.querySelectorAll<HTMLElement>('[data-code-variant]').forEach((el) => {
		el.hidden = el.dataset.codeVariant !== name;
	});
	return frame.querySelector(`[data-code-variant="${name}"]`);
}
