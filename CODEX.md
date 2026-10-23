# Codex agent: muse_attention_decode

You are the Codex agent. Your brief is CLAUDE.md in this repo (goal, rules, budget, environment,
workflow): read it, then TASK.md and AGENTS.md, and follow them, with these differences.

## Where you work

- Your clone is `~/codex-repo`, on the `codex` branch. Use `codex-exp/<name>` branches for risky work.
- Push only `codex` and `codex-exp/*` branches. Never push to `main`, never force-push. `main` belongs
  to the Claude agent working in `~/repo` on the same machine; don't touch that directory.
- Your log is `NOTES-codex.md` (newest at the bottom), not NOTES.md. Read NOTES.md too: it holds the
  history of this kernel, its measurements and its dead ends, so you don't repeat them.
- You may `git merge origin/main` to pick up the Claude agent's wins, but your value is a different
  angle, not a copy. Promising directions nobody here has tried yet:
  - a hand-written kernel (CUDA via `torch.utils.cpp_extension.load_inline`, or CuTe DSL if it is
    installed) with producer/consumer warps for the gathered FP8 rows, for the full-attention layers;
  - cutting launch and latency overhead on the sliding layers (weight 46 per load), e.g. one launch
    for attention and merge without the fused-merge's critical-path cost.
  Whatever you build must keep within TASK.md's rules (graph-capturable, no host syncs, every key,
  FP8 cache, shape-only branching, nothing cached from `data`).

## Judge budget

Your share: at most 8 checks and 3 scored runs per UTC day (see CLAUDE.md). Count them in
NOTES-codex.md. Submit only code from your branch that a check showed beating the current best board
result for tech-cosmos.

## Stopping

If `~/STOP` or `~/STOP-codex` exists, wrap up (notes, commit, push) and stop.
