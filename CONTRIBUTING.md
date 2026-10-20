# Contributing

Thanks for helping with the course. This guide covers how to run the project
locally and how to add chapters, kernels and simulations. The course plan is
in [`plan.md`](plan.md); section 2 describes the chapter template.

## Run the site

You need Node.js 24.

```bash
cd site
npm ci
npm run dev
```

The dev server prints a local URL. Pages reload when you save.

## Run the tests

Simulation model tests (Vitest):

```bash
cd site
npm test
```

Kernel correctness tests (pytest), from the repository root:

```bash
pytest kernels
```

You do not need a GPU for the kernel tests. `kernels/conftest.py` sets
`NUMBA_ENABLE_CUDASIM=1` and `TRITON_INTERPRET=1` for you when needed, so
Numba and Triton kernels run on the CPU. CUDA C++ kernels need `nvcc`; the
image in [`docker/`](docker/README.md) has it and everything else.

## Add a chapter

Chapters are MDX files in `site/src/content/docs/part-N-*/`, for example
`site/src/content/docs/part-2-memory/06-coalescing.mdx`. The sidebar is built
from these folders.

Frontmatter fields:

| Field | Meaning |
|---|---|
| `title` | Page title |
| `description` | One sentence, used for search and link previews |
| `chapter` | Chapter number as a string, for example `"6"` or `"0.1"` |
| `status` | `planned`, `draft` or `ready` |
| `prerequisites` | Chapter numbers to read first, for example `["2", "4"]` |
| `kernels` | Folder under `kernels/`, for example `ch06-coalescing` |
| `colab` | Notebook path under `notebooks/` |

```mdx
---
title: Global memory and coalescing
description: Why the order in which threads read memory decides bandwidth.
chapter: "6"
status: draft
prerequisites: ["2", "4"]
kernels: ch06-coalescing
colab: ch06-coalescing.ipynb
---
```

Follow the chapter structure in section 2 of `plan.md`. Open a chapter request
issue first if the chapter is not in the plan.

## Add a kernel

Each chapter's code lives in one folder:

```
kernels/chNN-slug/
├── cuda/                # .cu files and a Makefile with an `all` target
├── numba/               # .py
├── triton/              # .py
└── test_correctness.py  # checks every version against a NumPy reference
```

- All three versions compute the same result as a NumPy (or PyTorch)
  reference, and `test_correctness.py` checks each one.
- CI compiles every `kernels/*/cuda` folder with `make -C <folder> all`, so
  keep that target working.
- The site imports these files with `?raw`, so the page shows the same code
  the tests run. Keep them readable.

## Add a simulation

Each simulation has its own folder:

```
site/src/sims/<name>/
├── model.ts         # pure logic, no DOM access
├── model.test.ts    # Vitest tests for the model
├── view.ts          # SVG rendering
└── <Name>.astro     # wrapper that mounts the sim inside SimFrame
```

- `model.ts` holds all of the logic, so the numbers on screen can be tested
  without a browser. Test the cases a learner will try, such as the
  transaction count for stride 1 and stride 32.
- `view.ts` only draws the current model state as SVG.
- `<Name>.astro` wraps the sim in `SimFrame`, which provides the shared
  controls.
- Every sim must work with the keyboard alone, and must show a text status
  that describes each step, so the sim is usable without seeing the animation.
- Say what the model leaves out. Sims explain one idea; they are not cycle
  accurate.

## Writing style

- Use plain language and short sentences. Explain a term the first time it
  appears.
- Do not use em dashes.
- Verify every technical claim against the
  [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
  or the documentation of the library in question (Numba, Triton, PyTorch).
  Link the source in the PR description when the claim is not obvious.
- Do not quote performance numbers you have not measured. Measured numbers go
  in `kernels/*/results/` with the GPU they came from.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/),
for example:

```
feat(sims): add warp divergence simulation
fix(kernels): correct bounds check in ch06 transpose
docs(part-2): clarify shared memory bank width
```

Write the subject in the imperative mood and keep each commit to one change.

## Pull request checklist

- [ ] `npm test` and `npm run build` pass in `site/`.
- [ ] `pytest kernels` passes.
- [ ] New CUDA C++ code compiles with `make -C kernels/<chapter>/cuda all`.
- [ ] New sims work by keyboard and show a text status for each step.
- [ ] Technical claims are checked against the CUDA Programming Guide or the
      library docs.
- [ ] Commits follow Conventional Commits.

## Licensing of contributions

Code (kernels, simulation source, site code) is licensed under the MIT License
in [`LICENSE`](LICENSE). Course text, diagrams and simulation designs under
`site/src/content/` are licensed under CC BY-SA 4.0, see
[`LICENSE-content`](LICENSE-content). By opening a pull request you agree that
your contribution is licensed under the same terms as the files it changes.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
