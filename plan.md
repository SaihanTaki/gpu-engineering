# GPU Engineering: Interactive GPU & CUDA Learning Platform

## 1. Goal

A free, open-source, browser-based course that teaches GPU architecture and CUDA programming through
interactive simulations (SVG + HTML + JavaScript). Every chapter pairs a visual
model of the concept with working code in three languages:

- **CUDA C++**: the reference implementation, closest to the hardware
- **Numba (`numba-cuda`)**: CUDA's thread model, written in Python
- **Triton**: block-level programming, which is how most modern ML kernels are written

The learner should be able to *see* why a kernel is fast or slow before they
profile one on real hardware.

### Target learners
- Python/ML engineers who want to write custom kernels
- C++ developers new to GPU programming
- Students taking parallel computing courses

### Prerequisites
- Basic C/C++ or Python
- Arrays, pointers/indexing, big-O
- No GPU needed for the simulations. Code exercises run in the project's Docker
  image on a local NVIDIA GPU, on Colab/Kaggle, or on CPU in simulator/interpreter mode.

---

## 2. Chapter template

Each chapter follows the same structure:

1. **Hook**: a concrete problem ("why is this transpose 10x slower than a copy?")
2. **Concept**: short prose with static diagrams
3. **Simulation**: one or two interactive SVG widgets with play/pause/step,
   parameter sliders, and a code panel that highlights the current line
4. **Code**: tabbed `CUDA C++ | Numba | Triton` versions of the same kernel,
   with a "naive vs optimized" toggle where it applies
5. **Measure**: expected speedups, a roofline position, and what to look for in Nsight
6. **Exercises**: 2–4 tasks: predict-the-output, fix-the-bug, optimize-it
7. **Quiz**: 3–5 quick checks, graded in the browser
8. **Further reading**: links to the CUDA Programming Guide, papers, and blog posts

### Language coverage note
Triton deliberately hides threads and warps: you program a whole *block* of
data at once. For chapters about thread indexing, warps, divergence, shuffles,
and bank conflicts, the Triton tab explains **what Triton does for you
automatically** instead of showing an equivalent kernel. This contrast is a
lesson in its own right.

Legend for the tables below: ✅ full kernel · ◐ partial / explanation · — not applicable

---

## 3. Curriculum

### Part 0: Why GPUs?

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 0.1 | CPU vs GPU | Latency vs throughput, few big cores vs many small ones, Amdahl's law | **Race**: 8 fast workers vs 1000 slow workers on the same task; slider for the serial fraction | ✅ | ✅ | ✅ |
| 0.2 | The GPU software stack | Driver, runtime, PTX, SASS, nvcc, JIT; where Numba and Triton fit | **Compilation pipeline**: click a stage to see the code at that level (Python → Triton IR → PTX → SASS) | ◐ | ◐ | ◐ |

### Part 1: Foundations

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 1 | GPU architecture | SMs, CUDA cores, tensor cores, register file, L1/shared, L2, HBM | **Explorable chip**: zoom from die → GPC → SM → warp scheduler; pick a GPU preset (T4, A100, H100) | — | — | — |
| 2 | Programming model | Kernels, grid/block/thread, `threadIdx`/`blockIdx`/`blockDim`, 1D/2D/3D indexing | **Grid indexing explorer**: set grid/block dims, hover a thread to see its global index and which array element it touches | ✅ | ✅ | ◐ `program_id` + `tl.arange` |
| 3 | First kernel: vector add | Host vs device memory, `cudaMalloc`/`cudaMemcpy`, launch config, bounds checks, error handling | **Data journey**: arrays move CPU RAM → PCIe → HBM → SM → back, with a time bar showing transfer vs compute | ✅ | ✅ | ✅ |
| 4 | Execution model | Warps, SIMT, warp scheduling, latency hiding, divergence | **Warp scheduler**: warps stall on memory, the scheduler switches to ready warps. **Divergence**: 32 lanes going through `if/else`, masked lanes greyed out | ✅ | ✅ | ◐ |
| 5 | Occupancy | Registers/thread, shared mem/block, threads/block limits, achieved vs theoretical occupancy | **Occupancy calculator**: sliders for regs, smem, and block size; SM resources fill up visually | ✅ | ✅ | ◐ `num_warps`, `num_stages` |

### Part 2: Memory hierarchy

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 6 | Global memory & coalescing | Memory transactions, 32B sectors, 128B lines, strided and misaligned access, AoS vs SoA | **Coalescing visualizer**: a warp's 32 addresses mapped to memory segments; count transactions; pick a stride pattern | ✅ | ✅ | ◐ block pointers |
| 7 | Shared memory | `__shared__`, `__syncthreads`, tiling, data reuse | **Tile loader**: a block cooperatively loads a tile, then reuses it; global reads counted with and without shared memory | ✅ | ✅ | ◐ implicit |
| 8 | Bank conflicts | 32 banks, conflict degree, broadcast, padding, swizzling | **Bank grid**: 32 lanes → 32 banks; conflicts light up red; toggle `+1` padding | ✅ | ✅ | ◐ |
| 9 | Registers, caches & other memory | Register spilling, local memory, constant memory, read-only/texture cache, L1/L2 behavior | **Cache sim**: an access stream with hits and misses in L1/L2 | ✅ | ✅ | ◐ |
| 10 | Unified & pinned memory | Pageable vs pinned, Unified Memory, page faults, prefetch | **Page migration**: pages fault and migrate between CPU and GPU | ✅ | ✅ | — |
| 11 | Case study: matrix transpose | Naive → shared memory tiled → padded (conflict-free) | **Side-by-side**: three kernels with transaction counts and bandwidth bars | ✅ | ✅ | ✅ |

### Part 3: Parallel patterns

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 12 | Map & elementwise fusion | Kernel fusion, memory-bound ops, launch overhead | **Fusion**: 3 separate kernels vs 1 fused kernel; memory traffic counter | ✅ | ✅ | ✅ |
| 13 | Reduction | Tree reduction, sequential addressing, warp shuffles (`__shfl_down_sync`), grid-wide reduction, atomics | **Reduction tree**: step through levels; compare interleaved vs sequential addressing; idle threads shown | ✅ | ✅ | ✅ `tl.sum` |
| 14 | Scan (prefix sum) | Hillis–Steele vs Blelloch, work vs step efficiency, decoupled look-back | **Scan animator**: both algorithms side by side with operation counters | ✅ | ✅ | ✅ `tl.associative_scan` |
| 15 | Histogram & atomics | Atomic contention, privatization, shared-memory histograms | **Contention**: threads queue on one hot bin vs privatized copies | ✅ | ✅ | ◐ `tl.atomic_add` |
| 16 | Stencil & convolution | Halo cells, 1D/2D convolution, constant memory for filters | **Halo loader**: tile + halo regions; slider for filter radius | ✅ | ✅ | ✅ |
| 17 | Sorting | Bitonic sort, radix sort, merge | **Bitonic network**: the compare-swap network animated | ✅ | ✅ | ◐ |
| 18 | Sparse matrices | CSR/COO/ELL formats, SpMV, load imbalance | **Format explorer**: one matrix in each format; thread-to-row assignment and imbalance | ✅ | ✅ | ◐ |

### Part 4: Performance engineering

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 19 | Roofline model | Arithmetic intensity, memory vs compute bound, peak FLOPs/bandwidth | **Interactive roofline**: drag a kernel's FLOPs/bytes and watch it move; GPU presets | — | — | — |
| 20 | Matrix multiply, step by step | Naive → coalesced → shared-mem tiling → 1D/2D register blocking → vectorized loads → autotuning | **GEMM tiler**: tile sizes as sliders; data reuse heatmap; FLOPs/byte and est. GFLOPS for each step | ✅ | ✅ | ✅ (with `@triton.autotune`) |
| 21 | Streams & concurrency | Streams, async copies, overlapping transfer and compute, events, CUDA Graphs | **Timeline**: Gantt chart of H2D / kernel / D2H across 1–4 streams | ✅ | ✅ | — |
| 22 | Profiling & debugging | Nsight Systems, Nsight Compute, `compute-sanitizer`, reading metrics, common bugs (races, OOB) | **Race condition sim**: two threads interleave read-modify-write; step to see a lost update | ✅ | ✅ | ✅ `TRITON_INTERPRET=1` |

### Part 5: Modern GPUs & AI kernels

| # | Chapter | Key concepts | Simulation | C++ | Numba | Triton |
|---|---------|--------------|------------|-----|-------|--------|
| 23 | Tensor cores & mixed precision | FP32/TF32/FP16/BF16/FP8 formats, WMMA/MMA fragments, accumulation precision | **Bit layout explorer**: the same number in each format, rounding error shown. **MMA tile**: fragment ownership across a warp | ✅ WMMA | ◐ | ✅ `tl.dot` |
| 24 | Softmax & LayerNorm | Numerical stability, online softmax, row-per-block fusion | **Online softmax**: running max/sum updated chunk by chunk | ✅ | ✅ | ✅ |
| 25 | FlashAttention | Attention memory traffic, tiling Q/K/V, online softmax rescaling, IO-awareness | **Attention tiler**: blocks of Q×Kᵀ computed tile by tile; HBM vs SRAM traffic counters vs naive | ◐ | — | ✅ |

The course ends at FlashAttention. Quantization, multi-GPU/NCCL, CUTLASS/CuTe
and Hopper/Blackwell-specific features (TMA, wgmma, clusters) are out of scope
for v1.

### Appendices
- **A. PTX & SASS reading**: how to read `cuobjdump`/`nvdisasm` output
- **B. Environment setup**: Docker image for local GPUs (recommended), Colab, or a native CUDA toolkit install
- **C. Glossary**: every term links back to the chapter that introduces it
- **D. GPU spec sheet**: numbers used by the simulations (SM count, bandwidth, smem/SM, etc.)

---

## 4. Simulation design principles

- **Pedagogical, not cycle-accurate.** Each sim models one idea. A "simplified
  model" note lists what it leaves out.
- **Shared controls**: play / pause / step forward / step back / reset / speed.
  Every sim can be stepped so learners can pause on any state.
- **Code ↔ visual sync**: the code panel highlights the line that is executing
  in the animation.
- **Predict first**: where possible, ask the learner to guess (e.g. the number
  of transactions) before revealing the answer.
- **Parameters, not presets**: sliders for block size, stride, tile size, etc.,
  with sensible defaults and "interesting" presets ("try stride = 32").
- **Real numbers**: GPU presets (T4, A100, H100, RTX 4090) from Appendix D feed
  bandwidth/FLOP estimates.
- **Accessible**: keyboard controls, colorblind-safe palette, text alternatives
  for each animation state, works on mobile width.

---

## 5. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Site | **Astro** + **Starlight** (Astro's official docs theme), TypeScript | Ships zero JS by default; only the sims load scripts. Starlight provides the sidebar, prev/next, search, dark mode, mobile layout and "Edit this page" links. Deploy the `dist/` output to GitHub Pages |
| Theme customization | Starlight `customCss` for fonts and colors; component overrides (e.g. `Footer`, `PageTitle`) only where needed | Keeps Starlight upgradable; we restyle it instead of forking it |
| Chapter content | **MDX** (built into Starlight) in the `docs` collection (`src/content/docs/`), one folder per part. Starlight's default `docsSchema`; extend it only when a component needs a new field (e.g. a Colab link) | Prose and sims in one file; schema catches missing metadata at build time |
| Simulations | Plain **SVG + TS** classes, wrapped in `.astro` components that mount them from a `<script>` tag. **D3** only for scales/transitions | No UI framework needed; each sim's JS loads only on pages that use it |
| Shared sim library | `sim-core`: timeline/stepper, controls, code-highlight bridge, color tokens built on Starlight's CSS variables | Every sim behaves the same way and follows Starlight's light/dark theme |
| Code display | Starlight's `<Code>` component (Expressive Code, Shiki-based) inside Starlight `<Tabs syncKey="lang">` for C++ / Numba / Triton, wrapped in our `CodeTabs.astro` | Highlighting happens at build time. `syncKey` remembers the learner's language choice across every chapter |
| Code source | Kernel files imported from `kernels/` with Vite's `?raw` import | The page shows the exact file that the tests run |
| Navigation & search | Starlight sidebar autogenerated from each part's folder; prev/next links; built-in **Pagefind** search | Works offline, no search server, nothing to build ourselves |
| Running code | **Docker image** for local NVIDIA GPUs (see below), plus an "Open in Colab" notebook per chapter | Browsers can't run CUDA |
| No-GPU fallback | Numba `NUMBA_ENABLE_CUDASIM=1`, Triton `TRITON_INTERPRET=1` | Learners can run kernels on CPU for correctness |
| Quizzes/progress | Client-side, `localStorage` | No accounts in v1 |

### Docker image for local GPUs

One image with everything needed to build, run, test and profile every kernel
in the course:

- **Base**: `nvidia/cuda:<version>-devel-ubuntu24.04` (includes `nvcc`, CUDA
  headers, `compute-sanitizer`)
- **Python**: PyTorch (which brings Triton), `numba-cuda`, NumPy, pytest, JupyterLab
- **Tools**: CMake, Nsight Compute CLI (`ncu`), Nsight Systems CLI (`nsys`)
- **Host requirement**: an NVIDIA driver plus the NVIDIA Container Toolkit

```bash
docker build -t gpu-engineering docker/
docker run --gpus all -it --rm -p 8888:8888 -v "$PWD":/workspace gpu-engineering bash
# inside the container
make -C kernels/ch06-coalescing   # build the CUDA C++ version
pytest kernels/ch06-coalescing    # check all three versions against NumPy
python kernels/ch06-coalescing/bench.py
```

- Versions of CUDA, PyTorch, Triton and `numba-cuda` are pinned and listed on
  each chapter page, so the code a learner runs matches the course.
- Learners build the image locally; it is not published to a registry.
- Profiling inside the container needs `--cap-add=SYS_ADMIN` (or the host's
  performance-counter setting enabled); Appendix B documents both.

---

## 6. Repository layout

```
gpu-engineering/
├── plan.md
├── site/                      # Astro + Starlight project
│   ├── astro.config.mjs       # starlight() config: sidebar groups, editLink, customCss, base path
│   ├── tsconfig.json          # path aliases: @sims, @components, @kernels
│   ├── src/
│   │   ├── content.config.ts  # docsLoader() + docsSchema()
│   │   ├── content/docs/
│   │   │   ├── index.mdx               # landing page (Starlight splash template)
│   │   │   ├── part-0-why-gpus/        # 0-1-cpu-vs-gpu.mdx, 0-2-software-stack.mdx
│   │   │   ├── part-1-foundations/     # 01-architecture.mdx ... 05-occupancy.mdx
│   │   │   ├── part-2-memory/
│   │   │   ├── part-3-patterns/
│   │   │   ├── part-4-performance/
│   │   │   ├── part-5-ai-kernels/
│   │   │   └── appendix/
│   │   ├── styles/custom.css  # fonts, colors, sim tokens
│   │   ├── components/
│   │   │   ├── CodeTabs.astro          # Starlight Tabs + Code for the 3 languages
│   │   │   ├── Quiz.astro
│   │   │   └── overrides/              # Starlight component overrides, if any
│   │   ├── sim-core/          # stepper, controls, code-sync, theme (TS)
│   │   └── sims/              # one folder per simulation
│   │       ├── grid-indexing/
│   │       │   ├── GridIndexing.astro  # markup + <script> that mounts the sim
│   │       │   ├── model.ts            # pure logic, unit-tested
│   │       │   └── view.ts             # SVG rendering
│   │       ├── warp-divergence/
│   │       ├── coalescing/
│   │       └── ...
│   └── public/                # static assets
├── kernels/                   # runnable code, one folder per chapter
│   └── ch06-coalescing/
│       ├── cuda/              # .cu + CMakeLists.txt
│       ├── numba/             # .py
│       ├── triton/            # .py
│       ├── test_correctness.py
│       └── bench.py
├── notebooks/                 # Colab notebooks generated from kernels/
├── docker/
│   ├── Dockerfile             # CUDA devel + PyTorch/Triton + numba-cuda + Nsight CLIs
│   └── requirements.txt       # pinned Python versions
├── docs/                      # contributor guide, sim style guide
├── .github/workflows/         # site deploy, CPU tests, image build, GPU benchmarks
├── LICENSE                    # code license
├── LICENSE-content            # course text license
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

The code shown on the site is pulled from `kernels/` at build time, so the
displayed code is always the tested code.

A chapter then reads like this:

```mdx
---
# src/content/docs/part-2-memory/06-coalescing.mdx
title: 6. Global memory & coalescing
sidebar:
  order: 6
---
import { Aside } from '@astrojs/starlight/components';
import Coalescing from '@sims/coalescing/Coalescing.astro';
import CodeTabs from '@components/CodeTabs.astro';
import cuda from '@kernels/ch06-coalescing/cuda/copy.cu?raw';
import numba from '@kernels/ch06-coalescing/numba/copy.py?raw';
import triton from '@kernels/ch06-coalescing/triton/copy.py?raw';

A warp issues 32 loads at once...

<Aside type="tip">Try stride = 32 and count the transactions.</Aside>

<Coalescing stride={1} />

<CodeTabs cuda={cuda} numba={numba} triton={triton} />
```

Each sim keeps its logic in `model.ts` with no DOM access, so the model can be
unit-tested with Vitest separately from the SVG view.

---

## 7. Testing & quality

- **Correctness**: every kernel is checked against a NumPy/PyTorch reference
  (`pytest`), and CUDA C++ kernels against a CPU reference.
- **CI**: on CPU, run Numba in simulator mode and Triton in interpreter mode;
  build CUDA C++ with `nvcc` (compile only). Run real-GPU benchmarks on a
  self-hosted or cloud runner when available.
- **Sims**: Vitest unit tests for each sim's `model.ts` (e.g. the transaction count for a
  given stride) so the visuals match the math.
- **Benchmarks**: record numbers per GPU in `kernels/*/results/` and show them
  in the chapter.

---

## 8. Open-source project setup

- **Licenses**: MIT for code (kernels, sims, site) and CC BY-SA 4.0 for course
  text and diagrams. Using separate licenses lets people reuse the kernels
  freely while derived versions of the course text stay open.
- **Hosting**: public GitHub repo; the site deploys to GitHub Pages from `main`
  with GitHub Actions. The Docker image is built locally, not published.
- **Contributions**: `CONTRIBUTING.md` explains how to add a chapter, a sim or
  a kernel, using the chapter template from section 2. The sim style guide in
  `docs/` keeps contributed sims consistent.
- **Issue templates**: bug in a kernel, wrong explanation, new sim idea, chapter request.
- **Errata**: each chapter page has an "Edit this page" and "Report a problem"
  link to GitHub.
- **Pull requests**: CI must pass (CPU tests, site build, sim model tests)
  before merge; reviewers check technical accuracy against the CUDA docs.

---

## 9. Milestones

| Phase | Scope | Done when |
|---|---|---|
| **M0: Skeleton** | Starlight scaffold (sidebar groups per part, `custom.css`), `sim-core`, `CodeTabs`, one sim (grid indexing), Dockerfile, licenses, `CONTRIBUTING.md`, Pages deploy | Chapter 2 is fully readable and interactive |
| **M1: Foundations** | Part 0 + Part 1 (ch 0.1–5) | Learner can write and launch a kernel and explain warps/occupancy |
| **M2: Memory** | Part 2 (ch 6–11) | Transpose case study with all three languages and benchmarks |
| **M3: Patterns** | Part 3 (ch 12–18) | Reduction + scan + histogram sims shipped |
| **M4: Performance** | Part 4 (ch 19–22), GEMM is the flagship chapter | GEMM reaches a stated % of cuBLAS on A100/T4 |
| **M5: AI kernels** | Part 5 (ch 23–25) | FlashAttention forward pass in Triton with a tiling sim |
| **M6: Polish** | Glossary, search, progress tracking, accessibility pass | Public launch |

Suggested first build order inside M0/M1: grid indexing → warp divergence →
occupancy calculator → coalescing. These four sims cover most of the
`sim-core` features the later sims need.

---

## 10. Decisions

- **Audience & hosting**: public open-source course (section 8).
- **GPU access**: Docker image for local NVIDIA GPUs, plus Colab notebooks (section 5).
- **Scope of Part 5**: ends at FlashAttention.
- **Site theme**: Astro Starlight, restyled with custom CSS (section 5).

## 11. Open questions

- **AMD/ROCm**: add HIP equivalents later? (Triton already supports AMD GPUs.)
