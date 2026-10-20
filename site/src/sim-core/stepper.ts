/**
 * Drives a simulation through a fixed number of discrete steps.
 *
 * Sims compute their whole visual state from the step index, so the stepper
 * only tracks a number. That keeps "step back" exact and makes every state
 * reachable from the controls.
 */
export type StepListener = (step: number, stepper: Stepper) => void;

export class Stepper {
	/** Steps are 0..last. Step 0 is the initial state, before anything runs. */
	private _step = 0;
	private _last: number;
	private _playing = false;
	private timer: ReturnType<typeof setTimeout> | undefined;
	private listeners = new Set<StepListener>();

	/** Milliseconds between steps at speed 1. */
	baseInterval: number;
	speed = 1;

	constructor(last: number, baseInterval = 900) {
		this._last = Math.max(0, last);
		this.baseInterval = baseInterval;
	}

	get step() {
		return this._step;
	}
	get last() {
		return this._last;
	}
	get playing() {
		return this._playing;
	}
	get atEnd() {
		return this._step >= this._last;
	}

	/** Change the number of steps, e.g. after a parameter change. Resets to step 0. */
	setLast(last: number) {
		this.pause();
		this._last = Math.max(0, last);
		this._step = 0;
		this.emit();
	}

	goTo(step: number) {
		const next = Math.min(this._last, Math.max(0, Math.round(step)));
		if (next === this._step) return;
		this._step = next;
		this.emit();
	}

	next() {
		this.goTo(this._step + 1);
	}
	prev() {
		this.goTo(this._step - 1);
	}
	reset() {
		this.pause();
		this.goTo(0);
		this.emit();
	}

	play() {
		if (this._playing) return;
		if (this.atEnd) this.goTo(0);
		this._playing = true;
		this.emit();
		this.schedule();
	}

	pause() {
		if (!this._playing) return;
		this._playing = false;
		clearTimeout(this.timer);
		this.timer = undefined;
		this.emit();
	}

	toggle() {
		if (this._playing) this.pause();
		else this.play();
	}

	onChange(listener: StepListener) {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	private schedule() {
		this.timer = setTimeout(() => {
			if (!this._playing) return;
			this.next();
			if (this.atEnd) this.pause();
			else this.schedule();
		}, this.baseInterval / this.speed);
	}

	private emit() {
		for (const listener of this.listeners) listener(this._step, this);
	}
}
