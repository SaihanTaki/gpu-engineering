const SVG_NS = 'http://www.w3.org/2000/svg';

type Attrs = Record<string, string | number | boolean | undefined | null>;

/** Create an SVG element. `false`, `null` and `undefined` attributes are skipped. */
export function svg<K extends keyof SVGElementTagNameMap>(
	tag: K,
	attrs: Attrs = {},
	children: (SVGElement | string)[] = [],
): SVGElementTagNameMap[K] {
	const node = document.createElementNS(SVG_NS, tag);
	for (const [key, value] of Object.entries(attrs)) {
		if (value === undefined || value === null || value === false) continue;
		node.setAttribute(key, String(value));
	}
	for (const child of children) {
		node.append(typeof child === 'string' ? document.createTextNode(child) : child);
	}
	return node;
}

/** Remove all children of an element. */
export function clear(node: Element) {
	while (node.firstChild) node.firstChild.remove();
}

/**
 * Categorical colors for blocks, warps, etc. Defined in custom.css as
 * --sim-cat-0 .. --sim-cat-7 with light and dark variants.
 */
export const CATEGORY_COUNT = 8;
export const category = (i: number) => `var(--sim-cat-${((i % CATEGORY_COUNT) + CATEGORY_COUNT) % CATEGORY_COUNT})`;
