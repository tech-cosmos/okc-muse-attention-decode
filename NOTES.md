# Notes: muse_attention_decode

Newest entries at the bottom. Every session appends what it did, what it measured and what's next.

## State at handoff (2026-10-23, from the interactive session)

### Design (submission.py)

- `_attn_split`, grid (splits, B, Hkv/HPP): each program takes one split of one request's context for
  HPP KV heads. It normalizes (and on sliding layers ropes, via a [2^21, 64] cos|sin table built like
  SGLang's) its queries straight from `qkv`, streams paged FP8 K/V rows through an online softmax
  (bf16 MMA, exp2, scale folded), and writes partial (m, l, acc). Split 0 also normalizes/ropes the
  block's new keys, quantizes new K/V to FP8, stores them at `cache_loc`, and attends them causally
  from registers.
- HPP=2: both KV heads per program. Queries are block-diagonal [2*MG, 128] so one dot against the
  contiguous 128-byte cache row scores both heads (halves page-table lookups, full 128B row reads).
- `_combine`, grid (tokens, Hq): merges splits, RMSNorm, sigmoid(gate). Optional `fuse`: the last
  program of each (request, head pair) merges instead (atomic counter, reset after), one launch.
- `_plan(cfg, max_pages, device)` picks splits/block_n/warps/stages/hpp/fuse from shapes only.
  `tools/bench.py sweep` overrides it.

### Measurements

- H200 judge, check 6ffea83d (old 1-head kernel, untuned): 16/16 pass, 0.871x overall.
  sliding 1.2-1.9x faster than SGLang; full layers 0.57-0.79x (latency-bound gathers).
- H100 after HPP=2 + tuning: decode-32 full 302 us (SGLang on H200: 384 us; gather ceiling on H100:
  236 us). decode-1 sliding 6.8 us (SGLang H200: 20.9 us).
- Fused merge: slower at batch 1 (merge is on the critical path: 11.9 vs 6.8 us), 8-13% faster at
  batch 16-32 sliding with 8 splits.
- My kernel's H100 times were within a few % of its H200 judge times while latency-bound.

### Interpreter gotchas (tools/run.py, CPU-only)

Triton 3.8's CPU interpreter computes bf16 dots on raw bits and rounds fp32->e4m3 wrongly. run.py sets
`submission.DOT_DTYPE = tl.float32` and loosens the cache tolerance. On the H100 none of this matters.

### Ideas not yet tried

1. Apply the sweep's per-shape winners to `_plan` (results in sweep_all.log / below).
2. Verify blocks (q_len 8): M=64 rows per KV head; try FP8 QK^T (queries may be E4M3) to halve
   conversion work, or P@V in bf16 with V converted once per tile.
3. Full layers: persistent kernel / larger chunks so each program keeps more gathers in flight;
   prefetch the page table for the next tile explicitly; check SASS for cp.async on the gathers.
4. Sliding layers: fewer splits + fused merge at large batch; one CTA per (b, pair) at batch 32.
5. Combine kernel launch gap: consider PDL (programmatic dependent launch) if Triton exposes it.

### Budget

UTC day 2026-10-23: 3 checks and 2 scored runs used (17 checks, 8 runs left at 11:00 UTC).

## 2026-10-23 night (interactive session, continued alone on the H100)

### H200 judge results

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| 6ffea83d | first kernel | 0.871x | 16/16; full layers 0.57-0.79x |
| 8e51de3d | c0b75f0 (tuned plan, bf16) | **1.328x** (1.308-1.353) | 16/16; sliding 1.3-2.5x, verify-16/32 full 0.77-0.80x |
| 7037681c | e97cf37 (FP8 QK, FP16 PV, peel) | **1.652x** (1.627-1.677) | 16/16; H100 said 1.52x (H200 bandwidth helps full layers) |
| 0ae03dbe | 957e662 (scored run) | **1.807x** (1.780-1.834) | 16/16; verify 16/32 full 1.10x/1.25x, all else 1.44-2.43x |
| a9bf56a8 | e0378de (scored run) | **1.849x** (1.822-1.885) | 16/16; rank 18; sliding 1.49-2.86x, verify 16/32 full 1.09x/1.25x |

The H100 bench predicted 1.31x for c0b75f0, so it tracks the judge within a few %.

### What worked (H100 bench geomean vs SGLang's H200 times)

1.31x tuned plan -> 1.46x FP8 QK^T (E4M3 queries) -> 1.47x verify-full tiles -> 1.51x FP16 P@V
inputs (E4M3->FP16 is one cvt, BF16 needs two) -> 1.52x peeled tail tile (decode only).

### What didn't

- Fused last-program merge: never won in the sweep (on the critical path at batch 1).
- FP8 P@V (P*256 in E4M3): slower on every verify shape, errors up to 0.105 of 0.15.
- Peeling for verify blocks: +40% time (register pressure from two 64-row tile copies).
- `tl.range(..., warp_specialize=True)`: no effect on these gathered loads.
- HPP=2 for verify blocks: 128x128 fp32 accumulators, slow and slow to compile.

### Where verify full time goes (ablation, H100, verify 16/32 full = 202/404 us)

No exp: 193/390. No softmax at all: 187/382. So verify full is NOT ALU-bound; it is
latency/occupancy-bound (fat 64-row 4-warp programs keep few gathers in flight; decode moves the
same bytes in 251 us). Deeper pipelines help: 5 stages, 32-key tiles -> 188 us. Sweep running.

### Ideas next

1. Verify full: smaller programs (split the 8 query heads of a KV head across 2 programs, M=32)
   so more programs fit per SM; or prefetch the next tile's page indices one iteration early.
2. Decode full is ~6% above the random-gather ceiling on H100 (251 vs 236 us): little left there.
3. Sliding layers (weight 46): mostly launch + latency; check one-launch paths again at batch 32.

### Later the same night (H100 bench geomean vs SGLang's H200 times)

1.52x -> 1.56x deeper pipelines for verify (5-6 stages) -> 1.60x deeper pipelines for decode
(16-key tiles, 5-6 stages; 128-key tiles for batch 1) -> 1.63x one-pass merge over up to 128 splits
-> 1.68x programmatic dependent launch of the merge kernel (sliding layers 7-17% faster).
Then smoothed the plan between scored batches (no change on scored shapes; see nearby scan).

Dead ends: splitting a KV head's 8 query heads over 2 programs (duplicate K/V reads go to HBM, 2x
slower); 128-key tiles or 8 warps for verify full (10-15% slower); warp specialization.

Nearby-shape scan (batch 1-64, ctx 40K-120K, both layers, q_len 1/8): all correct, no cliffs after
the smoothing commit. Script: tools/nearby.py.

H200 per-shape (check 7037681c, e97cf37): verify 16 full 1.10x and verify 32 full ~1.04x are the
weak spots; everything else 1.39-2.50x. Verify full is a local optimum for this Triton design (it
is occupancy/latency-bound, see the ablation); the next step there is probably a hand-written
kernel with explicit producer/consumer warps (CuTe DSL or CUDA via load_inline).

## Board at overnight-agent start (2026-10-23 19:10 UTC)

We are #38 at 1.849x (a9bf56a8). #1 colinlikescode 2.366x, #10 2.069x, #20 1.956x; 340+ entrants.
To reach the top 20 we need ~6% overall; top 10 ~12%; #1 ~28%. Per-shape, our weak spots are verify
16/32 full (1.09x/1.25x on the H200): with 16/62 of each load's weight on full layers, those two
loads drag the geomean the most. Sliding layers (weight 46) are also worth re-examining: they are
launch/latency-bound, so shaving 1-2 us each there pays 46x.

Overnight agent: Daytona sandbox `okc-night` (H100, TTL 24h), started by tools/night.sh.

## Session 1 (overnight agent, 2026-10-23 19:10-~20:30 UTC)

Focus: verify full layers (verify 16/32 full were 1.09x/1.25x on the H200 and dragged their loads).

### Diagnosis (ncu on H100, verify 16 full, old kernel)

- 512 programs x 4 warps, 99 regs -> 4 programs/SM (register-limited), one wave; 16 warps/SM.
- Issue-active 62%, ~290 warp-instructions per 32-key tile; stalls spread over barrier/wait/
  long+short scoreboard. Not DRAM-bound (46% DRAM throughput). With L=8 every tile took the
  masked path (load masks + score masks).
- More pipeline stages (8-20) don't help; `maxnreg` 64-80 spills and is 2x slower.
- ncu works here: `/usr/local/cuda-13.0/bin/ncu` (scripts used: /tmp only; one-shot runner
  pattern = generate data via tools/bench.data_for, call kernel_fn 3x, `--launch-skip 2`).

### Changes (all on main, H100 bench geomean vs SGLang-on-H200)

1. 1.671 -> 1.699: scale folded into row max + `tl.fma` in exp2 (no per-score FMUL); tail tiles
   clamp the gather index to hi-1 instead of masking loads (only scores are masked).
2. -> 1.736: peel the masked tail for verify full too (now cheap: no load masks), 512 programs.
   (verify 32 full 379 -> 347 on H100, BUT on the H200 it went 311 -> 327: see check b42151ed.)
3. -> 1.731 (noise): verify sliding at batch 32: FP8 QK + peel (23.2 -> 22.3 us); full-verify
   splits floor 4 (batch 48/64 nearby shapes fill one wave).
4. -> 1.798: `dual`: a verify program runs BOTH KV heads as two independent 64-row softmax chains
   (two `_tile` calls per iteration; epilogue moved into `_finish`). 154 regs -> 3 programs/SM, but
   the two chains give ILP. Plus `balance`: each request's context is split into S equal
   tile-aligned chunks computed on device from its own seq_len (chunk = cdiv(cdiv(seq-base,S),N)*N),
   so programs = S*B*Hkv/2 all do equal work; S = 3*SMs // (B*Hkv/2) = exactly one wave
   (24 splits at batch 16, 12 at batch 32). verify 16/32 full: 179/347 -> 160/304 us on H100.
   Without balance the best split count (26) depended on the data's seq lens (empty trailing
   splits); with balance it's robust. Dual/balance lose on every other shape (tried decode full,
   verify 1 full, all sliding) so they're only on for verify full with B*Hkv > 4.
   Nearby scan (tools/nearby.py): all PASS, verify full now 116-129 ps/key from batch 8 to 64
   (was 135-166), no cliffs.

### Judge runs this session

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| b42151ed (check) | change 3 | **1.872x** (1.844-1.900) | 16/16; verify 16/32 full 1.19x/1.19x |
| 39d24ebc (check) | dcc2be4 (change 4, dual) | **1.898x** (1.870-1.927) | 16/16; verify 16/32 full 1.28x/1.29x (160/304 us) |
| 740bc5e9 (scored) | dcc2be4 (same code) | **1.921x** (1.892-1.950) | 16/16; best on board for us; verify 16/32 full 1.28x/1.28x |

H100 tracked the H200 closely for verify full under the dual design (160/304 on both). Sliding
layers and decode are unchanged and noisy at +-0.5 us between runs.

### Next ideas

1. Verify full is still ~2.5x the decode time for the same bytes (160 vs 101 us on the H200). Still
   issue/latency-bound. Ideas: cut per-tile instructions further (CS2R zeroing of the QK
   accumulator, V fp8->fp16 conversion going through smem with LDS/STS + fences each tile, 32 FMULs
   to rescale acc: try lazy rescaling only when the max grows by > 8 in log2 units, which needs a
   block-uniform branch); 3 chains per program (Hkv is only 2, so split keys within a program instead:
   two key-halves of the same head as independent chains, merged at the end).
2. Get dual under 128 regs -> 4 programs/SM (now 154 regs, 3/SM). Try stages 3-4 with BLOCK_N 64.
3. Verify sliding batch 32 (1.65x) and decode full (~2.0x) are the next weakest after verify full.
4. A dual chain for decode full (HPP=2 already reads full rows there; try 2 key-halves per program).

### Budget used (UTC day 2026-10-23)

This session: 2 checks (b42151ed, 39d24ebc) and 1 scored run (740bc5e9). Day totals: 5 of 15
checks and 3 of 3 agent-allowed scored runs (2 were the human's earlier), so no more scored runs today.
Board at 20:10 UTC still showed our old 1.849x (rank 43) right after the run; it may lag.

## Session 2 (overnight agent, 2026-10-23 20:15-~21:45 UTC)

Baseline reproduced: H100 bench 1.798x, 16/16 tests.

### Where the time goes now (H100 ablations, scripts were in /tmp; patterns described here)

- Verify full (dual): taking out the whole softmax only gets 161 -> 143 us (verify 16) and 306 -> 269
  (verify 32). Without exp: 152/289; without acc rescale: 155/296; without max update: 147/280.
  So no single softmax piece dominates; ~90% of the time is the gather pipeline itself (decode
  full moves the same bytes in 131 us on H100). Very sensitive to pipeline depth: stages 4 = 210 us,
  5 = 160 us.
- Sliding layers: a pure random 128B-row gather of 32 x 2048 rows (K+V, 16.8 MB) costs ~10 us on the
  H100 under a graph (minus a tiny-graph cost), so decode 32 sliding (16.5 us total, ~11.5 us of it
  in the KV loop) is already near the gather floor. Every decode-sliding config in a wide sweep
  (hpp 1/2, splits 4-32, warps 1-4, block_n 16-64) lands at 16.3-17.5 us.
- Sliding fixed costs (kernel with zero KV tiles): decode 3 us, verify 32 sliding 6.6 us (+4.8 us
  combine). RoPE adds ~1.3-1.6 us to the zero-tile kernel, but with tiles present making RoPE free
  (wrong-output ablation) only saves ~0.5-1 us on verify sliding and nothing on decode.
- _combine adds 1.3-4 us on sliding shapes (verify 32 sliding: 22.7 total vs 18.6 attn only).

### Tried, didn't help (all reverted)

- Lazy softmax rescale (FA4-style, rescale only when some row max grows > 8 in log2): 160 -> 184 us.
  The block-uniform branch needs a cross-warp max (smem + barrier) every tile.
- ex2.approx.f16x2 for p (inline asm): 161 -> 166 (the fp32 row sum needs a convert back).
- Window mask only on split 0's first tile (provably the only tile the window can cut for verify;
  never needed for decode since base = seq - 2047): ~12% fewer loop instructions, 0 time change.
  Sliding is latency-bound, not issue-bound (issue ~0.5/scheduler).
- RoPE pairs rotated in registers (reshape/split/join, contiguous cos/sin loads, no x[c^1] u16
  gathers): no change. Computing cos/sin in-kernel with libdevice (no 512 MB table load): far
  slower (verify 32 sliding 22 -> 34 us; overall 1.67x).
- Verify 32 sliding: fuse, splits 4/16, hpp 2, dual, block_n 64, 8 warps, maxnreg 128: all slower
  than the current plan (22.3 us).
- Verify full: warps 8, block_n 16/64, stages 3/4/6/8, maxnreg 96/104 (spills, 2.4x slower).

### Change (main)

- Verify full: `maxnreg=128` (no spills; the compiler used 160) lets 4 dual programs fit per SM.
  It wins only where 4/SM fills the single wave better than 3/SM (batch 8-24, 48: 2-3% faster) and
  loses where both leave the same 3% idle (batch 4, 32, 64), so `_plan` picks per/SM by wave fill.
  verify 16 full 160 -> 157 us; bench 1.798 -> 1.804. Correct at batch 8 (40K/120K), 12, 16, 20.

### Next ideas

1. Verify full is gather-pipeline-bound: more bytes in flight per SM without more registers.
   Options: K and V tiles for both heads as one 128-byte row load (dual reads 64B halves of the
   same rows in two loads; decode's hpp=2 reads full rows and is near the gather ceiling); prefetch
   the page-table slice of each split at program start; a hand-written producer/consumer kernel.
2. Sliding: fixed costs (prologue ~3-6 us, combine 1.3-4 us) are now the bulk above the gather floor.
   Ideas: issue the first page-table/KV loads before the q norm/RoPE (so the cos/sin trip overlaps
   the page-table trip); a smaller partial format is NOT allowed (keep fp32 partials).

### Judge runs this session

| job | commit | speedup | notes |
|---|---|---|---|
| 1985ba311f6b480a (check) | eedd003 (maxnreg 128 rule) | **1.896x** (1.868-1.924) | 16/16; verify 16 full 1.32x (was 1.28x: the H200 agrees), verify 32 full 1.28x; overall within noise of 39d24ebc/740bc5e9, so not worth a scored run |

### Budget used (UTC day 2026-10-23)

This session: 1 check (1985ba31), 0 scored runs. Under the new per-agent rule (CLAUDE.md, added mid-session:
8 checks + 3 scored per agent per UTC day, account 20/10 shared with the Codex agent; count both
agents' lines in ~/COORD.md), Claude has used 3 agent checks today (b42151ed, 39d24ebc, 1985ba31) and
1 scored run (740bc5e9); the human's earlier runs count against the account total. The day resets at 00:00 UTC.

Lanes (new): Claude = Triton submission.py on main; Codex = hand-written kernels on `codex` (it has
claimed a CUDA WMMA verify-full kernel with explicit gather prefetch). So "hand-written
producer/consumer kernel" in next idea 1 is Codex's lane now. Some of this session's early sweeps ran
before bench.py's GPU lock existed and may have overlapped Codex's runs; the final bench numbers above
were taken with the lock.

## Session 3 (overnight agent, 2026-10-23 20:52-~21:40 UTC)

Baseline reproduced: H100 bench 1.804x.

### Change (main 0495b16): grouped merge on sliding layers

`_combine_hb`: one program merges HB adjacent query heads of a KV head (their partial rows are
contiguous per split, so each split is one [HB, D] read). Sliding layers use hb=2, sb=8 splits per
pass, 1 warp; full layers keep `_combine` (no gain there, and worse at batch 1). Bench 1.804 -> 1.817:
verify 32 sliding 22.4 -> 20.7 us, verify 16 sliding 14.2 -> 13.4, decode 32 sliding 16.6 -> 16.2,
decode 1 sliding 6.3 -> 6.2. 16/16 tests, 48/48 nearby shapes (tools/nearby.py).

### Measured

- Merge share (total vs attn-only with `_combine` replaced by a no-op): decode 1 full 17.4/15.3,
  verify 1 full 19.0/16.0, decode 1 sliding 6.4/4.1, decode 16 sliding 11.6/9.6, verify 32 sliding
  22.7/17.4 (before the change), verify 16 full 156/149, decode 16 full 131/129. Script pattern:
  /tmp/parts.py (monkeypatch submission._combine with an object whose [grid] returns a no-op).
- Batch-1 full gather floor (80000 random 128B K+V rows from an 80K-row pool, under a graph, minus
  empty replay): ~12 us. Our attn kernel is 15.3 us there, so batch-1 full has only ~3 us + merge.
- Wide sweep of decode 1 full (splits 64-512, block_n 32-128, warps 2-8, stages 3/5, hpp 1/2): all
  17.3-18+ us. Flat.

### Tried, didn't help (reverted)

- `DISABLE_MMA_V3=1` (mma.sync instead of wgmma) for verify full: 2-2.6x slower.
- Interleaved dual tile (issue both heads' QK^T before either softmax): Triton still puts
  `warp_group_dot_wait {pendings = 0}` right after every wgmma (only loop-carried accumulator chains
  stay async), so no overlap; verify 16 full 156 -> 158-164, verify 32 305 -> 301 (noise).
- "Seq-free" batch-1 full loop (constant trip count, gathers depend only on the page table, clamped,
  page -1 -> row 0, scores masked by seq): no change (17.25 both). The seq-load trip is not the limiter.
- Grouped merge on full layers: equal at batch 16/32, 2-3 us slower at batch 1.

### Next ideas

1. The remaining merge cost on full verify 16 (~7 us of 156) is mostly partial traffic
   (17 MB of fp32 partials at 33 splits); fewer splits lose wave fill. Maybe hb>1 + PDL ordering.
2. Verify full in Triton is stuck on wgmma waits being synchronous; Codex owns the hand-written path.
3. Sliding prologue: the attn-only sliding times are still 2-3x the gather floor at batch 1.

### Judge runs this session

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| 944b20a9 (check) | 0495b16 | **1.935x** (1.906-1.969) | 16/16; verify 32 sliding 19.2 us (1.80x), verify 16 sliding 2.16x; full verify 1.32x/1.28x unchanged |
| 7d433848 (scored) | 0495b16 | **1.927x** (1.898-1.964) | 16/16; board best for us (was 1.921x, rank 32 before) |

### Budget used (UTC day 2026-10-23)

Claude today: 4 of 8 checks (b42151ed, 39d24ebc, 1985ba31, 944b20a9), 2 of 3 scored (740bc5e9, 7d433848).
Account before this session's runs: 6/20 checks, 3/10 scored. The day resets 00:00 UTC.
