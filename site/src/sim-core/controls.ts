import type { Stepper } from './stepper';

/**
 * Wires the static control bar rendered by SimFrame.astro to a Stepper.
 *
 * Keyboard (while focus is inside the sim, but not in a form field):
 *   Space / K  play or pause
 *   ← / →      step back / forward
 *   Home       reset
 */
export function bindControls(
	frame: HTMLElement,
	stepper: Stepper,
	describe: (step: number) => string,
) {
	const q = <T extends HTMLElement>(name: string) =>
		frame.querySelector<T>(`[data-sim-control="${name}"]`);

	const play = q<HTMLButtonElement>('play');
	const prev = q<HTMLButtonElement>('prev');
	const next = q<HTMLButtonElement>('next');
	const reset = q<HTMLButtonElement>('reset');
	const speed = q<HTMLSelectElement>('speed');
	const counter = q<HTMLElement>('counter');
	const scrub = q<HTMLInputElement>('scrub');
	const status = frame.querySelector<HTMLElement>('[data-sim-status]');

	play?.addEventListener('click', () => stepper.toggle());
	prev?.addEventListener('click', () => {
		stepper.pause();
		stepper.prev();
	});
	next?.addEventListener('click', () => {
		stepper.pause();
		stepper.next();
	});
	reset?.addEventListener('click', () => stepper.reset());
	speed?.addEventListener('change', () => {
		stepper.speed = Number(speed.value) || 1;
	});
	scrub?.addEventListener('input', () => {
		stepper.pause();
		stepper.goTo(Number(scrub.value));
	});

	frame.addEventListener('keydown', (event) => {
		const target = event.target as HTMLElement;
		if (target.closest('input, select, textarea')) return;
		switch (event.key) {
			case ' ':
			case 'k':
				event.preventDefault();
				stepper.toggle();
				break;
			case 'ArrowRight':
				event.preventDefault();
				stepper.pause();
				stepper.next();
				break;
			case 'ArrowLeft':
				event.preventDefault();
				stepper.pause();
				stepper.prev();
				break;
			case 'Home':
				event.preventDefault();
				stepper.reset();
				break;
		}
	});

	const sync = (step: number) => {
		if (play) {
			play.dataset.playing = String(stepper.playing);
			play.setAttribute('aria-label', stepper.playing ? 'Pause' : 'Play');
			play.title = stepper.playing ? 'Pause (Space)' : 'Play (Space)';
		}
		if (prev) prev.disabled = step === 0;
		if (next) next.disabled = stepper.atEnd;
		if (counter) counter.textContent = `Step ${step} / ${stepper.last}`;
		if (scrub) {
			scrub.max = String(stepper.last);
			scrub.value = String(step);
			scrub.setAttribute('aria-valuetext', `Step ${step} of ${stepper.last}`);
		}
		if (status) status.textContent = describe(step);
	};

	stepper.onChange(sync);
	sync(stepper.step);
	return sync;
}
