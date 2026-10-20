# Design direction

Owner answers (2026-10-20), transcribed as given:

- **Personality**: engineering lab. Hands-on and instrument-like; simulations feel like oscilloscopes or profiler panels.
- **Palette**: cool, teal/green accent.
- **Typography**: serif for body prose, sans for UI, simulations and controls.
- **Theme**: follow the OS setting, with a light/dark toggle.

Dial: ENERGY 2 / RHYTHM 2 / MOTION 1

MOTION 1 applies to site chrome (hover and focus states only). Simulations move only when the reader steps or presses play.

## Implementation choices

Proposed by the agent to fit the answers above. The owner can replace any of them.

| Decision | Choice | Reason |
|---|---|---|
| Body font | Source Serif 4 | Serif drawn for screen reading, with optical sizes for small and large text |
| UI font | IBM Plex Sans | Drawn for engineering and instrument use, which fits the lab personality |
| Code font | IBM Plex Mono | Same family as the UI font, so code and labels in sims line up visually |
| Font loading | Self-hosted via Fontsource | No third-party font requests; works offline in the Docker image |
| Accent | Teal (`#0f766e` light, `#5eead4` dark) | The owner's teal/green choice; used for the active step, active code line and primary actions only |
| Warning color | Amber | Marks problems in sims (out-of-bounds, uncovered elements); always paired with text or a pattern |
| Sim surfaces | Flat panels with a 1px hairline border, small radius | Reads as an instrument panel; no shadows or glow |
| Block colors in sims | Two alternating tints plus a printed block number | Adjacent blocks stay distinguishable without adding palette colors |
