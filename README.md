# okc-muse-attention-decode

Our entry for the Open Kernel Challenge problem `muse_attention_decode`: one decode step (or an
8-token verify block) of gated grouped-query attention over a paged FP8 KV cache, timed against
SGLang's kernel on an H200 under CUDA-graph replay.

Almost all of the code and notes here were written by coding agents (Claude Code and Codex) running
unattended in an H100 sandbox, steered through `CLAUDE.md`, `CODEX.md` and a shared coordination
file. The repo is published as it was at the end of that work, including the agents' logs and the
dead ends.

## Result

Best scored run on this repo: **2.542x** over SGLang (judge run `e64d3450`, 2026-10-25, rank 8 on
the board at that time). Per-shape H200 times from that run, sliding / full layer, in microseconds:

| shape     | sliding | full   |
|-----------|--------:|-------:|
| decode 1  |    4.44 |  13.57 |
| decode 16 |    7.53 |  87.66 |
| decode 32 |   11.93 | 168.19 |
| verify 1  |    5.74 |  16.18 |
| verify 16 |   11.90 | 106.19 |
| verify 32 |   15.42 | 196.60 |

## What's in `submission.py`

The kernel is dispatched by shape:

- **Triton** `_attn_split` + `_combine`: a split-context online softmax with FP32 partials, merged
  by a second launch chained with programmatic dependent launch. Used for sliding and small shapes.
- **Gluon** (Triton's low-level dialect) `_gw_verify` and `_gd_full`: warp-specialized full-layer
  kernels, with a producer warpgroup that gathers and converts the paged FP8 cache and consumer
  warpgroups that run the MMAs.
- **Native CUDA** kernels for the latency-bound sliding shapes, including a cluster kernel that
  merges through distributed shared memory.

The precision rules we held to: the cache stays FP8, split partials stay FP32, and nothing runs
below SGLang's precision.

## Layout

| path | what |
|------|------|
| `submission.py` | the kernel |
| `TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py`, `checks.py` | the challenge's own spec, baseline and checker, as written by `okc init` |
| `tools/bench.py` | the judge's 16 tests (`test`) and the 12 scored shapes under graph replay (`bench`) |
| `tools/sNN/`, `tools/a2/` | per-session probes and microbenchmarks |
| `STATE.md`, `NOTES.md` | the short current picture and the long session log (126 sessions) |
| `STATE-agent2.md`, `NOTES-agent2.md`, `COORD-final.md` | the second agent's notes and the shared coordination board |
| `CLAUDE.md`, `CODEX.md`, `AGENT2.md`, `tools/night.sh` | the instructions and loop the agents ran under |
| `profiles/` | per-kernel profiles of judge runs |
| `HANDOVER-lead.md`, `COORD-okc-night5-20261102.md` | the lead agent's handover and the coordination board from the later nights, which also cover the other OKC problems (decode layer, MoE, projection) |
| `post/`, `explainer-video.md` | the explainer video, leaderboard cards and the script that makes them |

Experiment branches (`exp/*`, `codex-exp/*`, `agent2-exp/*`) hold ideas that were built and
measured, most of which lost; `NOTES.md` has a dead-ends index.

## Running

Needs a Hopper GPU, torch 2.14.1+cu130 and triton 3.8.0 (see `tools/setup_sandbox.sh`).

```bash
python tools/bench.py test    # correctness, full size
python tools/bench.py bench   # the 12 scored shapes
```

`baseline.py` needs SGLang, which the sandbox did not have; SGLang's H200 per-shape times are
recorded in `tools/bench.py`.

## License

MIT for our code (see `LICENSE`). The challenge files listed above belong to the Open Kernel
Challenge organizers.
