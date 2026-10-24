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
- Precision: never below SGLang's. Split partials (accumulators and m/l) stay FP32, as FA3 keeps
  them; activations stay BF16 where SGLang has BF16. When in doubt, match SGLang's precision. (The
  human decided this on 2026-10-24 after BF16 partials had been used: don't reintroduce them.)
- Decisions may depend only on shapes and `config` (page_table.shape etc.), never on input values.
- Do not touch anything outside this repo and ~/okc (the venv). No new credentials, no other
  services. Don't delete the repo's history or force-push.

## Budget per UTC day (limits reset 00:00 UTC; check `okc problems --json` for what's left)

The account allows 20 checks and 10 scored runs per UTC day, shared by everyone using it. Two agents
work on this account (Claude on `main`, Codex on the `codex` branch; see CODEX.md), so each keeps to
its own share and counts its own runs in its notes file:

- `okc check submission.py --json`: at most 8 per UTC day per agent. Use them to confirm H100 wins on
  the real H200, not to explore.
- `okc submit submission.py --json`: at most 3 per UTC day per agent, and only when the H100 tests
  pass (`python tools/bench.py test`) AND a check run on the same code beat the best board result
  so far (`okc leaderboard muse_attention_decode` shows it; the other agent may have raised it).
  Leave the rest of the day's runs for the human.
- If `okc problems --json` shows fewer left than your share, the smaller number wins.
- Record every check/submit job id and its speedup in your notes file.

## Coordinating with the other agent (~/COORD.md)

`~/COORD.md` (in the sandbox's home, outside both repos) is the shared board between the two agents.
Ownership, so you don't duplicate each other: **Claude owns the Triton kernel in submission.py**
(plans, tiles, pipelining, merge, launches); **Codex owns hand-written kernels** (CUDA via
load_inline, CuTe DSL, inline PTX) on the `codex` branch. Stay in your lane unless COORD.md says the
other agent has handed a direction over.

- Session start: read ~/COORD.md. Under "Claims", add one line `<agent> <UTC time>: <what you will
  try this session>`. Don't take a direction the other agent claimed in the last 2 hours.
- Every check/submit you run: append a line under "Judge runs" (`<agent> <UTC time> <check|submit>
  <job> <commit> <speedup>`). Count today's lines from both agents against the account's 20/10.
- Session end: delete your claim and add one line under "Results" (what you tried, the measured
  numbers, branch/commit; dead ends too), so the other agent can use it.
- Re-read the file right before you write it and change only your own lines.
- Codex merges `origin/main` into `codex` to build on Claude's Triton work; Claude does not merge
  `codex`. A hand-written kernel that wins a shape gets dispatched by shape in Codex's submission.py.

## Sharing the GPU

The other agent uses the same H100. `tools/bench.py` takes a lock (/tmp/okc-gpu.lock) around each
timed replay loop, so timings stay clean; anything you time outside bench.py must take the same lock
(`fcntl.flock`). Expect benches to wait a little when the other agent is timing. Don't kill the other
agent's processes, and keep long sweeps to what you need.

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
