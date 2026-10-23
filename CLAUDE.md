# Overnight kernel work: muse_attention_decode

You are running unattended in a Daytona sandbox with one NVIDIA H100 80GB (Hopper, like the judge's
H200), working on the Open Kernel Challenge problem `muse_attention_decode` for the account
tech-cosmos. Each session is started by `tools/night.sh`; the previous sessions' notes are in
`NOTES.md`. Read TASK.md and AGENTS.md (the challenge's own spec and rules) before changing anything.

## Goal

Make `submission.py` faster than SGLang's kernel (`baseline.py`) on the 12 scored shapes, measured by
the judge as the geometric mean over 6 loads of SGLang's load time over ours (each load = 46 x sliding
layer + 16 x full layer). Leaders on the board are at about 2.37x. Real speedups only.

## Rules (non-negotiable)

- Follow TASK.md's rules and the spirit of the challenge: CUDA-graph capturable, no host syncs, no
  caching anything derived from `data`, FP8 cache stays FP8, queries no coarser than FP8 E4M3,
  attend to every key (no top-k/sparse/block-skipping), no tampering with timing or harness, no work
  outside the timed call, no reading harness frames, no processes or network from the submission.
  If something you did turns out to be reward-hacky, withdraw that run with `okc withdraw <job>` and
  say so in NOTES.md.
- Decisions may depend only on shapes and `config` (page_table.shape etc.), never on input values.
- Do not touch anything outside this repo and ~/okc (the venv). No new credentials, no other
  services. Don't delete the repo's history or force-push.

## Budget per UTC day (limits reset 00:00 UTC; check `okc problems --json` for what's left)

- `okc check submission.py --json`: at most 15 per UTC day. Use them to confirm H100 wins on the
  real H200, not to explore.
- `okc submit submission.py --json`: at most 3 per UTC day, and only when the H100 tests pass
  (`python tools/bench.py test`) AND a check run on the same code beat the best board result so far.
  Leave the rest of the day's runs for the human.
- Record every check/submit job id and its speedup in NOTES.md.

## Environment

- Python venv: `. ~/okc/bin/activate` (torch 2.14.1+cu130, triton 3.8.0, same as the judge image).
  No sglang here, so baseline.py can't run; SGLang's H200 per-shape times are in tools/bench.py.
- `okc` is on PATH and authenticated through `OKC_TOKEN`.
- `python tools/bench.py test`: the judge's 16 correctness tests at full size on the GPU (must pass).
- `python tools/bench.py bench`: the 12 scored shapes, CUDA-graph replay, cold L2, median, minus an
  empty replay. H100 times track the judge's H200 times closely for latency-bound shapes; bandwidth
  bound ones are ~1.4x faster on the H200.
- `python tools/bench.py sweep '<json>'` and `sweep_all`: time `_plan` overrides per shape.
- `python tools/gather.py`: the random-128B-row gather ceiling (~236 us on H100 for the batch-32 full
  layer's 655 MB; that's the floor to aim at for full layers).
- Long runs: run them in the background with output to a file and check on them; sessions are
  billed by time, so don't sleep idle for long.

## Workflow

1. Read NOTES.md (latest state, what was tried, what's next). Pick the most promising idea.
2. Work on a branch (`exp/<name>`) for anything risky; keep `main` = best known-good submission.
3. Every change: `python tools/bench.py test` must pass, then `bench`. Compare against the numbers in
   NOTES.md.
4. When a change wins: commit with a message that states the measured numbers, merge to main,
   `git push`, and append to NOTES.md.
5. End each session after one focused cycle (roughly 45-90 minutes of work): update NOTES.md (state,
   numbers, what you tried and learned, next ideas, budget used today), commit, push, and stop. The
   next session starts fresh from NOTES.md, so write it for a reader with no other context.
6. If `~/STOP` exists, wrap up and stop.
