export interface Question {
	/** Question text. `backticks` render as inline code. */
	q: string;
	/** Exactly four options. `backticks` render as inline code. */
	options: [string, string, string, string];
	/** Index (0 to 3) of the correct option. */
	answer: 0 | 1 | 2 | 3;
	/** Why the answer is right, shown after the reader picks. */
	why: string;
}
