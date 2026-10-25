# Overnight kernel work: muse_attention_decode

You are running unattended in a Daytona sandbox with one NVIDIA H100 80GB (Hopper, like the judge's
H200), working on the Open Kernel Challenge problem `muse_attention_decode` for the account
tech-cosmos. Each session is started by `tools/night.sh`; the previous sessions' notes are in
`NOTES.md`. Read TASK.md and AGENTS.md (the challenge's own spec and rules) before changing anything.

## Goal

**Be #1 on the board.** Score = the judge's geometric mean over 6 loads of SGLang's load time over
ours (each load = 46 x sliding layer + 16 x full layer). At 2026-10-24 18:00 UTC we were #13 at
2.343x (scored 7f45c5c6, main 14355d4) and #1 was 2.726x: we need ~16% more, so small plan tweaks
won't do it alone. Think big as well as small: new kernel structures, hand-written kernels (Gluon,
CUDA via load_inline, CuTe DSL, inline PTX are all in your lane now), fusing launches, and attacking
the weakest shapes first (H200: verify 16/32 full 1.79x/1.84x, verify 32 sliding 2.00x; the rest
2.2-3.5x). Check the board (`okc leaderboard muse_attention_decode`) at the start of each session.
Real speedups only.

## Current direction (set by the human's lead agent; overrides STATE.md's ranking)

Read ~/COORD.md's "Human decisions" section IN FULL at session start (it is at the TOP of the file, not the tail);
its newest entries are the current direction and outrank STATE.md. As of 2026-10-25 02:30 UTC:
1. Batch-1 shapes are OPEN and now first (s82 sensitivity table: decode/verify 1 sliding 1.3-1.6 %/us; decode/verify
   1 full 0.4-0.55 %/us, with ~3 us of serialized merge each). Native single-kernel designs (csl-style clusters,
   wgmma where rows allow, merges overlapped or fused) for verify 1 sliding, decode 1 full and verify 1 full. Read
   the dead-ends index first (fused last-program merges, cluster merges) and attack what made them lose.
2. Verify full only with a concrete structural idea (producer work or consumer latency at >= 16 warps/SM).
3. Physical-order decode 16/32 full (s80 design, H200 dense 144.7 vs random ~164 us): a multi-session native build;
   start only after a cheap prototype of the multi-split pass shows <= ~5 us.
4. Since 2026-10-25 04:15 UTC a second Claude agent (agent2, ~/repo2, branch agent2, AGENT2.md) owns batch 16/32
   sliding layers; leave those to it. At session start, check ~/COORD.md "Ready to merge": merge listed agent2
   commits into main (tests + bench), check, and submit per the usual rule. Leave >= 8 checks/day for agent2.
Do not revert this section; the lead agent edits it.

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
  P@V with P in FP8 E4M3 is at SGLang's precision (FA3's FP8 path, which SGLang runs here, feeds P to
  the P@V MMA as E4M3); confirm that in FA3's source before relying on it, and cite it in NOTES.md.
- Decisions may depend only on shapes and `config` (page_table.shape etc.), never on input values.
- Do not touch anything outside this repo and ~/okc (the venv). No new credentials, no other
  services. Don't delete the repo's history or force-push.

## Budget per UTC day (limits reset 00:00 UTC; check `okc problems --json` for what's left)

The account's daily limits are whatever `okc problems --json` reports (on 2026-10-24 it showed 20
checks and 10 scored runs). Codex is out of usage until 2026-10-30, so the Claude agent gets most of it:

- `okc check submission.py --json`: **all of the account's checks** (the human decided this on
  2026-10-24 while Codex is paused). Still spend them on real candidates. Checks are the scarce resource: the
  H100 does NOT always predict the H200 (batched-sliding hpp 2 won on the H100 and lost 0.8-1.1 us
  per shape on the H200). The judge reports per-shape times, so one check can test independent
  changes for different shapes at once: keep each change only for the shapes where it won.
- `okc submit submission.py --json`: at most 6 per UTC day, only when the H100 tests pass
  (`python tools/bench.py test`) AND a check run on the same code beat the board's best for
  tech-cosmos. Leave the rest for the human.
- If `okc problems --json` shows fewer left than your share, the smaller number wins.
- Record every check/submit job id, commit and per-shape result in NOTES.md and ~/COORD.md.
- **Out of checks is not a reason to stop.** Never end a session as a no-op: research and prototype
  on the H100 (bigger ideas especially), and queue candidates in STATE.md for the next day's checks.

## Coordinating with the other agent (~/COORD.md)

(Codex is paused until 2026-10-30; until then the lanes below don't bind and hand-written kernels
are yours too. Keep logging judge runs and results in ~/COORD.md.)

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

1. Read STATE.md first: a short (<= 150 lines) current picture you keep up to date (best scored run
   and its per-shape H200 times, what's on main vs checked, queued candidates for checks, ranked
   ideas, a dead-ends index pointing into NOTES.md). NOTES.md is the long log (2,600+ lines): grep
   it, don't read it whole. If STATE.md is missing, write it from NOTES.md first.
2. Work on a branch (`exp/<name>`) for anything risky; keep `main` = best known-good submission.
3. Every change: `python tools/bench.py test` must pass, then `bench`. Compare against the numbers in
   NOTES.md.
4. When a change wins: commit with a message that states the measured numbers, merge to main,
   `git push`, and append to NOTES.md.
5. End each session after one focused cycle (roughly 45-90 minutes of work): update STATE.md and NOTES.md (state,
   numbers, what you tried and learned, next ideas, budget used today), commit, push, and stop. The
   next session starts fresh from NOTES.md, so write it for a reader with no other context.
6. If `~/STOP` exists, wrap up and stop.
