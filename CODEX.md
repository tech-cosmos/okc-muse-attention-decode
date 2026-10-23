# Codex agent: muse_attention_decode

You are the Codex agent. Your brief is CLAUDE.md in this repo (goal, rules, budget, environment,
workflow): read it, then TASK.md and AGENTS.md, and follow them, with these differences.

## Where you work

- Your clone is `~/codex-repo`, on the `codex` branch. Use `codex-exp/<name>` branches for risky work.
- Push only `codex` and `codex-exp/*` branches. Never push to `main`, never force-push. `main` belongs
  to the Claude agent working in `~/repo` on the same machine; don't touch that directory.
- Your log is `NOTES-codex.md` (newest at the bottom), not NOTES.md. Read NOTES.md too: it holds the
  history of this kernel, its measurements and its dead ends, so you don't repeat them.
- Ownership (see "Coordinating" in CLAUDE.md, and ~/COORD.md): you own **hand-written kernels**
  (CUDA via `torch.utils.cpp_extension.load_inline`, CuTe DSL if installed, inline PTX); the Claude
  agent owns the Triton kernel. Don't tune Triton plans; it does that. Merge `origin/main` into
  `codex` regularly so you build on its latest Triton kernel, and dispatch your kernel by shape
  where it wins, so `codex` is always at least as fast as `main`.
- Directions, roughly in order of payoff:
  - full-attention layers (verify 16/32 full are the weakest shapes, ~1.1-1.3x on the H200): a
    gathered-FP8 flash-decode kernel with producer/consumer warps (cp.async or TMA gather into
    shared memory, several pages in flight per warp), close to the ~236 us random-gather ceiling;
  - sliding layers (weight 46 per load, launch/latency-bound): one launch for attention + merge.
  Whatever you build must keep within TASK.md's rules (graph-capturable, no host syncs, every key,
  FP8 cache, shape-only branching, nothing cached from `data`). Compiling at import time is setup,
  not timed work, but keep it fast and robust (the judge must be able to build it).
- The judge runs in `lmsysorg/sglang:dev-muse-spark-open`, which has PyTorch, Triton, FlashInfer,
  DeepGEMM, TileLang, CuTe DSL and CUTLASS; native CUDA goes through `load_inline` (nvcc). A run has
  15 minutes in all, compiling included. The sandbox has CUDA 13.0 at /usr/local/cuda-13.0; install
  anything else you need into ~/okc only at versions matching that image.

## Judge budget

Your share: at most 8 checks and 3 scored runs per UTC day (see CLAUDE.md). Count them in
NOTES-codex.md. Submit only code from your branch that a check showed beating the current best board
result for tech-cosmos.

## Stopping

If `~/STOP` or `~/STOP-codex` exists, wrap up (notes, commit, push) and stop.
