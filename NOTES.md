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

## Session 4 (overnight agent, 2026-10-23 21:50-~22:50 UTC)

Baseline reproduced: H100 bench 1.80x (verify 16/32 full 156/304 us). No change kept; main unchanged
(0495b16 code). Board at ~22:45: #1 pjmullin3 2.726x, #2 2.366x; us #33 at 1.927x.

### Load arithmetic (where the geomean is, from check 944b20a9)

Per load (46 x sliding + 16 x full, H200 us, ours vs SGLang): decode 1 592/1469 (2.48x), decode 16
2150/4426 (2.06x), decode 32 3705/7593 (2.05x), verify 1 657/1628 (2.48x), verify 16 3108/4628 (1.49x),
verify 32 5740/7796 (1.36x). If verify full ran at decode-full speed (105/195 us on the H200), the
geomean would be ~2.16x. On batch-1 loads the sliding layer is ~58% of the load (46 x 7.5 us), so
-2 us on batch-1 sliding = ~+5% overall.

### Tried, didn't help (all reverted)

- "Wide" dual for verify full: one 128-byte K/V row gather for both heads, queries zero-padded to
  [64, 128] (QK^T twice as wide), V split per head with reshape/permute/split: 156 -> 217-346 us.
- Page indices one tile ahead in registers (loop-carried, so the K/V gathers don't wait on an
  in-loop page load): verify 16 full 156 -> 153.7 us at stages 3, but verify 32 full 304 -> 310. Noise
  level / no win. Side finding: Triton allocates only 3 K buffers and 2 V buffers for verify full at
  num_stages=5 (2 each for decode), so "stages" isn't the prefetch depth one would think.
- Polynomial exp2 (FA4-style, FMA pipe) on one or both dual chains: 156 -> 178/197 us. Verify full is
  issue-bound, not MUFU-bound.
- Fused last-program merge for batch-1 sliding (one launch): it was slow because the old code
  broadcast each split's weight through smem with 2 barriers per split (32 serialized L2 trips,
  visible in SASS) and the `.cg` loads cost more again. Rewriting it as a [SB, rows, D] reduction
  (like _combine_hb), dropping `.cg` (the acq_rel atomic orders the stores anyway) and preloading the
  gate at program start: decode 1 sliding 11.6 -> 7.0 us, but the current two-launch PDL path is
  6.2 us. Verify 1 sliding fused: 12+ us vs 7.1. A single SM merging 64 KB of partials is on the
  critical path; the atomic-only cost (no merge) was 4.7 us, so the merge would have to fit in 1.5 us.
- Fixed softmax shift (no running max) is NOT safe: cached keys are N(0,1) input data, not
  norm-bounded, so the score range isn't bounded by the rmsnorm. Don't revisit.

### Next ideas

1. Verify full is issue-bound (~290 warp-instructions per 32-key tile per warp). Count SASS per tile
   by category (cvt for V, ldmatrix/sts for the V fp16 staging, wgmma fences, shuffles) and attack the
   largest; the V fp8->fp16 staging through smem per tile per chain is a candidate.
2. Batch-1 sliding: the merge kernel is ~2 us of 6.2. Codex has a native warp merge for B1/2
   (codex branch, check 2d178f1d 1.942x); don't duplicate.

### Budget used (UTC day 2026-10-23)

This session: 0 checks, 0 scored runs (nothing beat main). Claude today: 4 of 8 checks, 2 of 3 scored.

## Session 5 (overnight agent, 2026-10-23 22:01-~22:45 UTC)

Baseline reproduced: H100 bench 1.802x. Two wins, both from the same lens: count the serial memory
round trips on the critical path of the latency-bound shapes (batch-1 and sliding layers).

### Changes (main)

1. 651b421, one-pass sliding merge: `_combine_hb` took 8 splits per pass (`sb=8`), so batch-1 sliding
   (32 splits) paid 4 serial L2 round trips. Now `sb = splits` (one pass, still 1 warp, hb=2).
   H100 1.802 -> 1.861: decode 1 sliding 6.27 -> 5.23, verify 1 sliding 7.39 -> 6.43, decode 16
   sliding 12.03 -> 11.41 us.
2. 65ff1ab, new tokens first: split 0 used to attend the block's own tokens (qkv k/v loads, cos/sin,
   cache_loc, FP8 store) AFTER its KV loop: one more serial HBM trip on the slowest program. Moved to
   before the loop (`_new`; `_finish` now only writes partials). Online softmax is order-free.
   H100 1.861 -> 1.911: decode 1 sliding 5.23 -> 4.77, verify 1 sliding 6.43 -> 5.55, decode 16
   sliding 11.41 -> 10.29, verify 16 sliding 13.42 -> 12.61 us; full layers equal or slightly better.
   A/B per shape (flag, since removed): new-first wins or ties everywhere; batch-1 full within 0.1 us.
   16/16 tests, 48/48 nearby shapes (tools/nearby.py) after each change.

### Measured (H100, after both changes)

- Merge share (total vs attn-only, /tmp/c5/parts.py pattern = monkeypatch `_combine`/`_combine_hb`
  with a no-op): decode 1 sliding 4.83/3.46, verify 1 sliding 5.82/4.32, decode 1 full 18.8/15.4,
  verify 1 full 19.0/16.3, decode 16 sliding 10.35/8.74, decode 32 sliding 15.6/14.2, verify 16
  sliding 12.6/10.5, verify 32 sliding 20.5/16.9 us.
- Equal seq_lens experiment (all set to the min, time scaled to the mean): verify 16 full no gain,
  verify 32 full 2.5%, decode full none. So per-request imbalance under `balance` costs <= 2.5%:
  global (cross-request) load balancing is not worth building.
- Verify-full SASS (dual, maxnreg 128): ~404 instructions per 2-chain 32-key iteration: 68 FMUL
  (acc rescale), 40 FMNMX, 40 FADD, 36 MUFU, 36 FFMA, 32 F2FP, 5 BAR.SYNC, 3 MEMBAR.ALL.CTA + proxy
  fences (V fp8 -> LDS -> F2FP -> STS fp16 -> fence -> bar -> HGMMA, both chains share ONE fp16 V
  staging buffer, so they serialize there), Q re-read by LDSM every iteration (register cap). The
  cp.async wait is DEPBAR.LE 5 with 5 groups/iteration (prefetch depth ~1 iteration, but an iteration
  is ~2 us so that's not the limiter). ~45% issue utilization: stall-bound on bars/wgmma waits.

### Tried, didn't help (reverted)

- Token groups for verify full (block's 8 tokens split over adjacent programs so duplicate K/V gathers
  meet in L2; 1-warp mma.sync programs like decode): 331-930 us vs 156. Dead.
- hpp=2 for verify full with 8 warps (one warpgroup per head, one 128B row gather, one V conversion
  per tile): 279 us (159 regs, 1 program/SM); maxnreg 128 -> 2/SM: 208 us at stages 5-8. Dead
  (wasted half of the MMA columns and of the acc rescale).
- Batch-1 sliding with 1-4 splits + fused merge (no merge launch): 11.5-29 us. One program can't
  hide serial gather latency.
- L2 prefetch of the whole page-table row (inline `prefetch.global.L2`) before the seq load, batch 1
  sliding: no change (and slower at batch 16).
- One-trip merge (load the accumulators together with the maxima when one pass covers all splits):
  bench 1.895 vs 1.911, no gain (the compiler already overlaps them, or noise).
- Re-sweeps after the changes (all flat or worse): sliding splits 8-64, block_n 32/64, warps 1-8,
  stages 2-6, merge hb 1-8 / cw 1-4 at every batch.

### Next ideas

1. Same lens elsewhere: list every dependent memory trip per program (seq -> page -> KV; seq ->
   cos/sin; partial store -> grid end -> merge loads) and look for ones that can be issued earlier.
2. Sliding merge is still 1.4-3.5 us. Verify 32 sliding: 2048 one-warp merge programs over 8.4 MB of
   partials.
3. Verify full: stuck at ~156/305 us in Triton (see SASS above); Codex measured native variants slower.

### Judge runs this session

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| 554277594e264b05 (check) | 651b421 | **1.960x** (1.931-1.998) | 16/16; decode 1 sliding 6.21 us (3.16x), verify 1 sliding 7.35 (2.99x) |
| 254b1d40063f429c (check) | 65ff1ab | **2.041x** (2.011-2.072) | 16/16; decode 1 sliding 6.23 (3.33x), verify 1 sliding 7.04 (3.23x), decode 16 sliding 10.25 (2.62x), verify 16 sliding 12.51 (2.31x); full layers 1.28-2.17x |
| 0a2eec0fdfad40af (scored) | 65ff1ab | **2.021x** (1.991-2.051) | 16/16; board best for us, rank 21 (was 33 at 1.9456x) |

H200 weak spots now: verify 16/32 full 1.33x/1.28x, verify 32 sliding 1.83x; everything else 2.05-3.33x.

### Budget used (UTC day 2026-10-23)

Claude today: 6 of 8 checks (b42151ed, 39d24ebc, 1985ba31, 944b20a9, 55427759, 254b1d40), 3 of 3 scored
(740bc5e9, 7d433848, this session's). Account before this session's submit: 12/20 checks, 5/10 scored.

## Session 6 (overnight agent, 2026-10-23 23:19-~23:58 UTC)

Baseline reproduced: H100 bench 1.893x (decode 1 sliding 4.90, verify 1 sliding 5.87 us). No change kept;
main unchanged (65ff1ab code). Board at 23:55: #1 pjmullin3 2.726x, us #21 at 2.021x (#20 2.036x).

### Measured (scripts in profiles/session6/)

- One dependent HBM round trip (cold L2, pointer chase under a graph, 64 programs) = ~0.45 us; an empty
  launch is ~0 after the empty-replay subtraction (trips.py).
- ncu per-instruction stall sampling works and is the right tool for the latency-bound shapes:
  `ncu --kernel-name _attn_split --launch-skip 2 --launch-count 1 --section SourceCounters
  --warp-sampling-interval 0 --import-source no --page source --csv --print-source sass python one.py
  "<shape>" '<plan json>'`, then `top.py <csv> 0.012` lists the hot SASS lines with their stall reason.
- Decode 1 sliding _attn_split (main), share of stall samples: ~10% `imc` on the first ULDC (the kernel's
  constant bank is a cold DRAM miss after the L2 flush, every kernel pays it, unavoidable), ~6% seq wait,
  ~2% cos/sin wait, ~11% page-table wait (DEPBAR + barrier), ~15% K/V gather wait. SASS order on main:
  seq -> q/cos loads -> q math -> split 0's `_new` (its loads and math) -> THEN the loop's page-table
  LDGSTS. So the page trip does not overlap the cos/sin trip, and in split 0 not `_new`'s trip either.
- Ablations (attn only, decode 1 sliding, us): all 3.58; no `_new` 3.17; no q RoPE 3.44; no KV loop
  1.92; fixed seq 3.30. Removing the new-token cache stores alone: no change (so store/load aliasing is
  not what holds the loop back).

### Tried, didn't help (reverted; diff of the last variant in profiles/session6/first_tile.diff)

- New-token FP8 stores moved after the KV loop (recomputed there): bench 1.893 -> 1.832.
- Issuing the first tile's page + K/V loads before the q prologue (register loads, then the tile's
  softmax by hand): decode 1 sliding attn-only 3.42 -> 3.97, worse on every shape. ncu: K/V landing in
  registers and then staged to smem for the MMA (STS) waits about twice as long as the pipelined
  cp.async gathers (847 vs 440 samples), and ptxas put the STS stall before the q math anyway.
- Same, K/V loads after the q math: still worse (3.36 -> 3.68).
- Only the first tile's page indices loaded early (fed into the cp.async loop through a masked in-loop
  page load + where): decode 1 3.57 -> 3.52, verify 1 4.53 -> 4.45, decode 16 9.18 -> 9.71. ptxas then
  ordered the cos/sin load behind the page load. Not worth it. Lesson: in Triton we can't pin the SASS
  load order; overlapping the page trip with the cos/sin trip would need inline PTX or a hand-written
  kernel (Codex's lane).
- Sliding with few wide programs (splits 4-16, block_n 128/256, warps 4/8, stages 1/2): all slower
  (decode 1 5.4+, verify 1 6.6+, decode 16 11.1+). Stages 1/2 at batch-1 sliding: stages 2 looked
  0.2 us faster on verify 1 in one A/B, but 7-rep medians were equal (5.66 vs 5.63). Noise.

### Next ideas

1. Batch-1 sliding is a chain of ~5 serial trips (imc, seq, cos, page, KV, plus `_new` in split 0) at
   ~0.45 us each, plus barriers. The only Triton-level way I see left to cut one is to take the q math
   off the path before the loop (e.g. compute q inside the first loop iteration, after the pipeline
   prologue has issued the page/K/V loads). That's untested and fiddly.
2. Verify full (1.28-1.33x on H200) is still the biggest lever (80%+ of the verify 16/32 loads); see
   sessions 2-5 for what has been ruled out in Triton.

### Budget used

This session: 0 checks, 0 scored runs. UTC day 2026-10-23 for Claude: 6/8 checks, 3/3 scored. The day
resets at 00:00 UTC; 2026-10-24 starts with Claude 0/8 checks, 0/3 scored.

## Session 7 (overnight agent, 2026-10-23 23:38-23:50 UTC)

Baseline reproduced: H100 bench 1.876x (verify 16/32 full 155.6/306.7, decode 1 sliding 5.02 us). No change
kept; main unchanged (65ff1ab code). 0 checks, 0 scored runs.

### Tried, didn't help (all reverted)

- Verify full, V one tile ahead in registers (plain loads, page indices two tiles ahead, loop-carried,
  dual chains): verify 16/32 full 156/305 -> 311-322/503 us; with maxnreg 168 (3/SM) still 197/382.
  SASS: LDS count unchanged (110 vs 112) and PRMT 60 -> 136: Triton converts the loaded blocked fp8 V
  layout through shared memory anyway (STS/LDS byte shuffles) before the FP16 staging, so nothing is
  saved, and registers go 148 -> 210. Dead.
- Verify full, both KV heads in one block-diagonal product (`_attn_bd`): K/V tiles [2N, 128] loaded as the
  same cache rows twice with per-half masks (cp.async zero-fill), q [64, 128] = both heads' queries side by
  side, S = [64, 2N] with per-half row max/sum via where-masks, one P@V into acc [64, 128] with a per-half
  alpha. One V staging per tile instead of two. 257/512 us (N=16/head), 446 (N=32). Lots of SEL/PRMT/LOP3;
  doubled V conversion work. Dead (like session 4's wide dual).
- Batch-1 sliding, q built inside the KV loop (each program has exactly one tile; `start < 0` dependency to
  stop LICM, split 0 builds its own q in `_new`): decode 1 sliding 4.8 -> 5.6, verify 1 sliding 5.8 -> 12.2.
  Same lesson as session 6: Triton/ptxas load ordering can't be steered this way.
- Batch-1 full merge variants (_combine_hb hb 2/4, sb 32-128, cw 2-8): all 17.97-19.4 vs 18.0 (flat).
  Merge share there: decode 1 full 17.7 total vs 15.7 attn-only, verify 1 full 19.4 vs 16.2.
- Decode 16 full re-sweep (stages 6/8, splits 48/96, block_n 32): current plan best (130.5 us).
- ncu stall sampling on verify 32 sliding: spread over gather waits (long_sb at the barrier/fences before
  wgmma), wgmma waits and barriers; no dominant line. imc ~2%, end drain ~2%.

### Where things stand

The Triton design looks locally optimal on every scored shape; seven sessions of sweeps and restructures
are flat or worse. Remaining gaps: verify full ~1.25x the decode-full time for the same bytes (issue-bound
softmax and V staging; Codex's native attempts so far are slower than Triton there), batch-1 merges
(~2-3 us full, ~1.4 us sliding; Codex has a native batch-1 full merge on `codex` that saves ~1 us), and
batch-1 sliding serial trips. All three point to hand-written code (Codex's lane). Ideas still open in
Triton: none I'd rate above noise. Maybe revisit if Triton gains async wgmma for non-loop-carried dots or
explicit load ordering.

### Budget

Claude: 0 checks / 0 scored this session. UTC day 2026-10-23 ended at 6/8 checks, 3/3 scored; 2026-10-24
starts at 0/8, 0/3.

## Session 8 (overnight agent, 2026-10-23 23:50 - 2026-10-24 ~00:40 UTC)

Baseline reproduced: verify 16/32 full 155.4/305.4 us on H100.

### Change (main 57ad70c): verify full at 4 programs/SM from 6 pairs up

Session 2's rule (4/SM only where it fills the wave better than 3/SM) no longer holds after the later
changes: with maxnreg 128 + 4/SM, batch 32 verify full goes 303 -> 292 us, batch 40 397 -> 358, batch 64
593 -> 568, batch 6/8/16 0.5-4 us faster, batch 12 equal; only batch 3-4 lose (44.7 -> 45.5, 52.5 -> 54.7).
Rule now `per = 4 if pairs >= 6 or fill[4] > fill[3] else 3`. Bench 1.905 (verify 32 full 296 us).
16/16 tests, 48/48 nearby shapes, batch 32/64 verify full also checked against the reference.

### Tried, didn't help (reverted)

- P straight in FP16 from `ex2.approx.f16x2` (inline asm, pack=2; half the MUFU ops, no F2FP after),
  row sum as (a) fp32 tl.sum, (b) a dot with a ones [N,16] matrix accumulated on the tensor cores,
  (c) pairwise fp16 add then fp32 sum: verify 16 full 155.4 -> 161.4 / 170.0 / 156.1, verify 32 305.4 ->
  325.8 / 329.2 / 314.7. Also with 4/SM: 302-303 vs 292. MUFU is not the limit.

### Budget (UTC day 2026-10-24)

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| fdccc1c6fa704926 (check) | 57ad70c | **2.013x** (1.983-2.053) | 16/16; verify 32 full 303.9 -> 295.4 us (1.283x -> 1.321x), verify 16 full 156.4 (1.316x, unchanged); overall within noise of 254b1d40 (2.041x) / scored 0a2eec0f (2.021x), so no scored run |

The okc limit for this problem now reads 10 checks/day (not 20). Claude on 2026-10-24: 1 check, 0 scored.

### Next ideas

- Verify full is still ~1.3x SGLang on the H200 and the biggest lever. In Triton, sessions 1-8 ruled out
  softmax-side cuts (MUFU, rescale, exp form); the limit is issue utilization (~45%) at 16 warps/SM with
  the register cap. The occupancy/wave-fill rule is the one knob that still moved; maybe re-sweep splits
  around 4/SM (e.g. 2 waves of smaller chunks) at batch 16/32.

## Session 9 (overnight agent, 2026-10-24 00:32-~01:00 UTC)

Baseline: main 57ad70c unchanged (H100 verify 16/32 full 154.7/294-296 us). No change to submission.py; 0 checks,
0 scored runs.

### Re-sweeps (no win)

- Verify full at 4/SM (maxnreg 128), splits 16-66: the current one-wave count is best (verify 16: 33 -> 154.8,
  30/40/44/50/66 -> 166-194; verify 32: 16 -> 295.3, 33 -> 302, 18-25 -> 341-382). Two waves of smaller chunks lose.
- Verify full block_n 64 (stages 2-5, with/without maxnreg): 207-291 us vs 155. Dead.

### New direction: Gluon (triton.experimental.gluon, ships with triton 3.8), branch exp/gluon-verify

Gluon is Triton's low-level dialect: explicit layouts, shared-memory allocation, cp.async rings with explicit
wait_group, async wgmma with explicit `warpgroup_mma_wait(n)`, mbarriers, warp_specialize. It removes the two
Triton limits the earlier sessions hit (wgmma waits pendings=0 after non-loop-carried dots; no control of the
gather ring). Prototype: tools/gluon_verify.py on exp/gluon-verify (commit 507f50e), cached keys only (no new
tokens, no merge), writes the same partials as _attn_split; bit-identical to Triton's partials for splits >= 1.

H100, verify 16 / verify 32 full partials (Triton main: 155 / 294 us whole kernel_fn, ~149 / ~287 attn only):

| variant | regs, smem, per SM | verify 16 | verify 32 |
|---|---|---|---|
| single head per program, 64-byte half-row gathers, serial waits, S=24 | 74, 28 KB, 6 | 160.4 | - |
| same, S=16/20/33 (wave fill matters: S*32 programs vs 6*132 slots) | | 182/169/215 | |
| same + FA3-style QK(i+1) during softmax(i) (88 regs, 5/SM), S=20 | | 172 | |
| same + PV wait deferred to next iteration (88 regs) | | 238 (S=24) | |
| same + page indices 1-3 tiles ahead | 75-77 | 166-171 | |
| same, BN 64 / maxnreg 64-80 | | 200-233 / 214-462 | |
| dual: both heads, one 128-byte row gather for K and V, zero-padded [64,128] queries, NS=4, 130 regs (3/SM) | | 198 (S=24) | |
| dual, maxnreg 128, NS=3 (4/SM) | | 175 | |
| dual + page indices 2 tiles ahead | | 170 | |
| dual + per-head K slice (`k_tile.slice(h*64, 64, 1)`, [64,64] queries, half the QK work and Q smem), NS=4, pf | 128, 56 KB, 4 | **154.0** | **304.4** (S=16) |
| same, single fp16 V staging buffer (VHB=1) | 128, 48 KB | 155.0 | 304.4 |
| same, NS 5/6 (spills) or BN 16 (maxnreg 96-128) | | 173-315 | |

So the best Gluon kernel is at parity minus ~3-6% with Triton's attn kernel; not integrated. ncu on the best
variant: stalls spread (V fp8->fp16 conversion short_sb, barriers, wgmma waits); before the page-index lead the top
stall (14-17%) was the page-table load feeding the gather address. Register cap 128 (4/SM) is the hard limit:
NS >= 5 spills.

Gluon gotchas (cost time, write down):
- Casting an fp32 tensor that came from a bf16 load and was zeroed with where/masked-load to float8e4nv produced
  NONZERO bytes (0x60, 0xFC) for the zeros in a [64,128] BlockedLayout([1,8],[4,8],[4,1]) tensor (repro:
  `/tmp/g9cv3.py` pattern: load bf16 [64,64] twice as [64,128] via c % 64, where(c < 64, x, 0), .to(fp8)). Pure
  fp32 input converts fine. Workaround: zero AFTER the cast (`where(first, x8, zeros_like(x8))`), or avoid zero
  padding (per-head slices). The Triton submission's decode hpp=2 block-diagonal queries cast zeros too, but the
  tests pass there.
- Register operands of an async wgmma (P for P@V) must be passed in `warpgroup_mma_wait(deps=[...])` or the
  compiler may reuse the registers while the MMA reads them.
- NVMMA smem slices along the contiguous dim at 64 bytes inside a 128B swizzle work as wgmma operands.
- `maxnreg` works as a launch kwarg on gluon kernels; `kernel.device_caches[0][0]` holds the compiled kernels
  (n_regs, n_spills, metadata.shared).

### Next ideas

1. Gluon verify full is within ~5% of Triton with a design that is still simple. Things that could flip it:
   (a) warp_specialize with a 1-warp producer (pages + cp.async + `async_copy.mbarrier_arrive`, noinc, ready count
   32) and an mbarrier empty ring, so the consumer warpgroup has no address math or page waits; the catch is
   occupancy: 160 threads x 128 regs > 4/SM, so use worker_num_regs (setmaxnreg) with the producer at 24-40 regs
   and launch maxnreg ~100. (b) V fp8->fp16 conversion of tile i+1 done right after issuing PV(i) (needs tile i+1
   landed: wait_group(NS-3)). (c) Fewer barriers: the cp.async-ready barrier and the V-staged barrier could merge
   if V conversion moves one tile ahead.
2. If Gluon wins on verify full, integrate as a shape-dispatched path in submission.py (new tokens + _combine
   unchanged; the Gluon kernel only replaces the cached-key loop; split 0's new-token work could stay in a tiny
   Triton kernel or move into Gluon). Check COORD lanes first (Gluon is Triton-family; noted in COORD as Claude's).

### Budget (UTC day 2026-10-24)

Claude: 1 check (fdccc1c6, session 8), 0 scored. This session: none. Account at 00:35: 3/10 checks, 0/10 scored.

## Session 10 (overnight agent, 2026-10-24 01:00-~02:00 UTC)

### Change (main a82d996): verify full from 17 pairs up runs a warp-specialized Gluon kernel

`_gw_verify` (submission.py, Gluon, guarded by `HAVE_GLUON`), FA3 style, per (split, request, KV head pair):
- worker partition, 4 warps at 24 regs (`_gw_producer`): page indices (next tile's loaded before waiting for the
  slot), cp.async of the pair's 128-byte K and V rows into an NS=4 slot ring of [64, 128] fp8 tiles,
  `cp.mbarrier_arrive(ready[s], increment_count=False)` (ready count = 128 producer threads);
- two consumer warpgroups (default partition + one worker, 96 regs), one per KV head: wait ready, FP8 QK^T wgmma
  on its 64-byte half (`slice(HEAD*D, D, 1)`), V half -> FP16 smem, wait QK, `mbarrier.arrive(empty[s])` (count 2),
  online softmax, P@V wgmma;
- launch maxnreg 80 x 384 threads -> 2 CTAs/SM (Triton pads the 4-warp worker to a warpgroup; .reqntid 384);
  setmaxnreg gives the partitions their counts. BN 64 tiles (BN 32 was 157 us, BN 64 145 us partials-only).
- splits = (4*sms - 1) // pairs (verify 16: 32, verify 32: 16; one more split hits a wave-quantization cliff).
- New tokens: a small Triton kernel `_new_tokens` (grid B x Hkv, reuses `_new`/`_finish`) writes partial slot S
  and the FP8 cache rows; it triggers PDL at its start, the Gluon grid is launched with launch_pdl=True and runs
  concurrently (it reads only cached rows < seq), and executes `griddepcontrol.wait` (inline asm) at its very end so
  the merge (`_combine`, PDL, S+1 slots) sees slot S. Doing the new tokens inside the Gluon grid lost: as extra
  programs (+1 split each, 158-172 us at verify 16) or at the end of split 0's consumers (163 us).

H100 (Gluon vs Triton, verify full, ctx 80K agentic): batch 8 91.2/86.0, 12 125.4/122.5, 16 154.6/154.6,
24 218.3/223.9, 32 274.5/294.8, 40 335.4/360.5, 48 407.7/422.3, 64 524.0/577.4. Dispatch: pairs > 16.
Bench 1.905 -> **1.939** (verify 32 full 295 -> 275 us). 16/16 tests; with the Gluon path forced on every
verify-full test shape plus page 64 / kv_scale / dp / batch 17-64 / ctx 1M cases: 11/11 pass; CUDA-graph replay with
fresh inputs copied in (3 seeds, batch 32): pass.

Gotchas: Gluon has no gdc builtins -> `gl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=gl.int32,
is_pure=False, pack=1)`. A 1-warp producer gets padded to 4 warps anyway and at 24 regs spills (BN 64: 16 cp.async per
thread); PW=4 doesn't. tp8 has Hkv=2 -> pairs = batch (I mis-sized splits once by assuming 2 pairs).
Prototypes: tools/gluon_ws.py (single-consumer `_gws` and per-head `_gws2`), tools/gluon_verify.py (session 9).

### Follow-up (main 0a22b94): producer converts V

`_gw_producer_cv` / `_gw_consumer_cv` (plan `lag=1`): the producer warpgroup (40 regs) issues tile i's gathers, then
`wait_group(LAG)` + partition barrier, converts tile i-LAG's V halves to FP16 into per-head [NS, 64, 64] rings, fences,
arrives `ready` (count 1; Gluon's mbarrier.arrive is elected after a partition-local bar.sync). Consumers only do
QK^T, softmax, P@V, then arrive `empty` after P@V. NS 3 (32 KB/stage, 104 KB smem for 2/SM). H100: verify 32 full
275 -> 272, batch 24 218 -> 214, 64 524 -> 517, verify 16 ~153-155 (still a tie with Triton, so dispatch stays at
pairs > 16). BN 32 + NS 5-6 with LAG 2-3: 310 us (dead). NS 4 + LAG 2 at BN 64: 1 CTA/SM, 314 (dead).

### Judge (UTC day 2026-10-24)

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| 82a996b129e34019 (check) | a82d996 | **2.062x** (2.031-2.093) | 16/16; verify 32 full 295.4 -> 264.9 us (1.32x -> 1.47x on H200); everything else unchanged |

Not submitted: board best at the time was Codex's 2.058x (71e2eaa7), within noise, and Codex's pending check
b4853b2a is a82d996 + their native small-batch kernels (a superset). If that fails, a82d996 (checked) qualifies.
0a22b94 is unchecked (~1% on verify 32 full only).
Claude 2026-10-24: 2 checks (fdccc1c6, 82a996b1), 0 scored.

### Next ideas

1. Verify 16 full is still ~1.32x on the H200 and Gluon only ties there: at 2 CTAs/SM with S=32 it's 528 slots for
   512 programs. Ideas: 3 CTAs/SM with 1 consumer warpgroup per CTA doing one head (no pair sharing of rows, 64-byte
   gathers) + 1 producer warp... (Triton pads workers to warpgroups: 256 threads x 3 = 768, 85 regs each -> consumer
   ~120, producer 24 might fit); or ping-pong two consumer warpgroups on one head (FA3 intra-head pipelining).
2. Decode full (L=1, M=8 rows) could use the same producer/consumer structure with mma_v2 or a 64-row tile holding
   8 heads x 8 ... (all query heads of a pair: G=8 rows per head -> M=16 for 2 heads, wgmma needs 64: waste). Hard.
3. The merge costs ~8.7 us on the Gluon path vs ~6.5 on Triton's; the extra slot and the PDL chain may be the
   reason. Merge in the last-finishing program (atomic counter) was a loss before at small batch, maybe not here.

## Session 11 (overnight agent, 2026-10-24 02:27-~03:20 UTC)

### Changes on main

1. a794ac6: Gluon verify full (`_gw_verify`) from **6 pairs up** (was > 16), with **one wave** of splits:
   slots = 2 CTAs/SM x 132; splits = slots // pairs, or two waves only if they fill > 5% better (batch 40).
   The old (4*sms-1)//pairs gave ~2 waves of half-size programs. H100 verify full, ctx 80K agentic
   (old plan -> new): B4 52.4 -> 51.8 (stays Triton: pairs < 6), B6 69.4 -> 67.8, B8 86.6 -> 82.3,
   B12 123.7 -> 116.3, B16 155.2 -> 144.7, B20 181.5 -> 175.5, B24 213.3 -> 206.4, B32 270.6 -> 266.7,
   B40 332 (two waves kept), B48 398.5 -> ~400, B64 522 -> 513.
2. 824e52a: consumer masks keys only on a chunk's last tile (`gpeel`, `if lo + (i+1)*BN > hi`):
   verify 16 full 145 -> 139, verify 32 full 270 -> 259. Bench 1.939 -> **1.960**. 16/16 tests and 13
   forced-Gluon cases (/tmp/s11/force.py pattern: page 64, kv_scale, dp, edges, ctx 1M, batch 2-64) pass.

### Measured / dead ends

- Ablations on the Gluon verify kernel (H100, verify 16 / 32 full; base 138.8 / 259.9):
  no V fp8->fp16 conversion 137.1 / 254.5; no gathers (cp.async skipped, compute only) 106.2 / 201.9.
  So verify full is now mostly gather-bound: verify 16 full (139) is close to decode 16 full (130.5),
  which reads the same bytes. The full layers are all near the random-row gather floor now.
- Row sums on the tensor cores (P @ ones [BN, 8] wgmma instead of gl.sum): 139.4 vs 139.4. Dead.
- Cross-request balanced split (programs take equal slices of the batch's concatenated keys; segment
  (p, b) -> slot p + b; merge `_combine_bal` from the same prefix sums), branch exp/gluon-bal: correct,
  140.4 / 260.0 vs 139.3 / 258.9 (agentic seq max/mean is 1.06, but early finishers just leave
  bandwidth to the rest). P 248-528 swept. Dead.
- 1 CTA/SM with deep rings (splits 8, NS 5-6, LAG 1-2): 151-156 vs 139. BN 32 at one wave: 158. Dead.
- Scored-case page tables are fully random (0% consecutive), so no locality to exploit.

### Judge (UTC day 2026-10-24)

| job | commit | speedup (95% CI) | notes |
|---|---|---|---|
| 2363a04971ab46b1 (check) | 824e52a | **2.130x** (2.098-2.162) | 16/16; H200 verify 16 full 155.4 -> 129.6 us (1.32x -> 1.59x), verify 32 full 264.9 -> 240.3 (1.47x -> 1.62x) |
| d0f30dc0d435455d (submit) | 824e52a | **2.113x** (2.081-2.156) | 16/16; board rank 18 (was 2.091x, Codex's fe512312) |

Codex's check 3e786669 (a794ac6 + their native small-batch kernels) was 2.107x, so 824e52a + native should do better still.
Claude 2026-10-24: 3 checks, 1 scored. Account at 03:40: 8/10 checks, 4/10 scored.
Note: don't wrap `okc check` in `timeout 1100`; the queue can be > 15 min and the client then dies with
no output (the job still runs: find it with `okc jobs` / `okc status <job>`).

### Next ideas

1. Full layers are near the gather floor (decode 32 full 244 vs ~236 floor on H100; verify 32 260).
   What's left there is small: verify 16 full 139 vs decode 16 130.
2. The biggest remaining gaps per load are now the verify loads' full layers on the H200 (compute share
   is larger there since bandwidth is 1.4x) and the sliding layers (Codex's native lane at small batch).
   On the H200 verify 16/32 full are 129.6/240.3 us vs decode 97.8/185.3, so verify is compute-bound again there: the compute-only ablation (106 / 202 us on
   H100) is the bound to attack (consumer issue rate: softmax ops/element; e.g. FA4-style conditional
   rescale with a warp-uniform vote via inline asm, or exp2 emulation for part of the tile).

## Session 12 (overnight agent, 2026-10-24 03:33-~04:00 UTC)

No change to main (still 824e52a's kernel). 0 checks, 0 scored. Experiments on branch exp/gluon-pipe (df6bd0b,
all behind plan flags that default off); scripts in tools/s12 there (forced-Gluon tests, no-gather ablation copy,
ncu SASS-mix and spill/regs readers).

### Where the Gluon verify-full time goes (H100, verify 16 full, base 139 us)

- No gathers (cp.async removed, everything else kept) 106 us; minus V conversion 99; minus exp 101; minus P@V 94;
  no softmax at all 71. No merge kernel at all: 133 (verify 32: 254 vs 259), so the merge tail is ~6 us.
- ncu, no-gather variant: issue slots 50% busy, 1.1 eligible of 5.5 warps/scheduler. Per score element ~1.1 each of
  FADD (row sum), FMNMX (max), FMUL (acc rescale; D == BN so it's 1/element), FFMA, MUFU, plus 1.0 F2FP (P and V).
  Top stalls: wgmma waits (DEPBAR, ~9%), shuffle-reduction chains (short_sb on FMNMX/FADD), producer waiting on
  `empty`. Consumers retry the `ready` wait ~2x per tile. It is latency-bound, not issue- or MUFU-bound.
- Register split under warp_specialize: the DEFAULT partition (consumer of head 0) gets
  (maxnreg * 12 warps - 4 * sum(worker regs)) / 4, not CREGS. Base cregs 96 / pregs 40 => head 0 gets 104, head 1 96.
  Balanced is cregs 104 / pregs 32 (same speed as base). Read regs/spills with tools/s12/spills.py.

### Dead ends (do not retry at 2 CTAs/SM)

| variant | verify 16 full (H100) | compute-only |
|---|---|---|
| base | 139.1 | 105.6 |
| FA3 pipelining (QK(i+1) issued before softmax(i)), cregs 112/pregs 32 | 145.6 (96-104: 157-178) | 113.5 |
| same at 1 CTA/SM (cregs 200, splits 8, NS 5, LAG 2) | 149.2 | 129.4 (unpipelined: 141) |
| P@V(i) and QK(i+1) issued together, one wait | 171-233 (spills at <= 104 regs) | 160 |
| FA4 conditional rescale (skip acc*alpha unless a row max grows > 2^8; WG-wide vote) | - | 116.7 |
| FP8 QK^T into an FP16 accumulator (HMNMX2/HFMA2, ex2.approx.f16) | 144.0 (verify 32 267.8 vs 259) | 121-122 |
| same + row sums as P @ ones on the tensor cores | - | 114.5 |
| row sums as P @ ones alone (f32 path) | - | 106.3 (tie) |

Why FP16 scores lose: Hopper has no packed f16x2 MUFU (ex2.approx.f16x2 becomes two MUFU.EX2.F16) and the
half-packing costs ~2 PRMT per element; total instructions went up (48.5M -> 51.6-60.6M).

Decode full deeper pipelines (more bytes in flight, in case the H200 needs them): all slower on H100
(decode 32 full: stages 7/8 251 vs 243; block_n 32 249-251; splits 128 265). Can't tell the H200 effect without a check.

### Next ideas

1. Verify full: the consumer can't hold a second S tile at <= 104 regs. Only remaining compute lever I see is fewer
   live registers per consumer (e.g. M=64 split as two m64n32 halves with separate maxima, or acc in the producer's
   freed regs) so FA3 pipelining fits; or a native CUTLASS/FA3-style kernel (Codex lane).
2. The merge tail (~6 us of verify full, ~4%): FP16 normalized partials (acc / l) would halve its bytes and the main
   kernel's partial writes; needs _gw_store, _new_tokens and _combine to agree. Maybe 2-3 us.
3. H200-only effects (bytes in flight) are invisible here; a check comparing deeper decode-full pipelines would tell,
   but only worth it with spare checks.

Budget (UTC day 2026-10-24): Claude 3/8 checks, 1/3 scored (unchanged this session). Account at 03:45: 8/10 checks,
4/10 scored.

## Session 13 (overnight agent, 2026-10-24 03:50-04:20 UTC)

### Change (main 19b8ade): lighter merge on the Gluon verify-full path

`_verify_gluon` merges with `_combine_hb` (2 heads per one-warp program, all slots in one pass) when splits + 1 <= 32
(plan keys hb=2, cw=1; batch >= 12 at ctx 80K). H100 paired runs: verify 16 full 139.0 -> 137.3 us (repeated 3x),
verify 32 full 258.7 -> 258.3; nearby batch 12/16/20/24/32/48/64: -2.7/-1.3/-0.7/-1.1/-0.8/-1.7/-2.0 us; batch
6/8 (34+ slots) keep `_combine`. Bench 1.960 -> 1.979 (the rest is noise). 16/16 tests; new tools/gluon_cases.py
(8 shapes that reach the Gluon path, which none of the judge's 16 tests do: batch 8-64, page 1/64, kv_scale, edges,
mixed/agentic): 8/8 pass. Not checked on the H200 (expected ~+0.2% overall, inside the judge's CI; account had 2
checks left today).

### New tool: tools/emu.py (H200 per-SM bandwidth emulation)

Parks a spinner kernel (1024 threads, 200 KB smem, one per SM) on 40 SMs and times graph replays on the other 92
(H100 3.35 TB/s / 92 SMs ~= H200 4.8 TB/s / 132 SMs per SM), with `_num_sms` = 92 for planning.
`python tools/emu.py 40 full ['{"plan": ...}']`. Gotchas: CUDA lazy module loading deadlocks against the spinner
(every kernel must be loaded before it starts: the script sets CUDA_MODULE_LOADING=EAGER and warms up); never call
torch.cuda.synchronize() while it spins. It does NOT reproduce H200 times (total bandwidth is still the H100's); it
shows whether a kernel keeps HBM busy with H200-like per-SM demand. Watch wave quantization: decode plans don't scale
splits with SMs, so a config with fewer programs/SM can fall into two waves at 92 SMs only (scale splits by 92/132
to compare).

Measured with it (decode 32 full; 132 SMs / 92 SMs): main 243.5 / 260; pure random gather (tools/gather.py kernel)
236 / 241.6 at its best config (BN 64, 2 warps, 3 stages), but 237 / 263 at BN 32 x 1 warp x 4 stages. So at H200
per-SM rates our decode-full kernel loses ~7% that a wide-gather kernel doesn't, plausibly part of why H200 decode 32
full is 185 us (~74% of 4.8 TB/s). Every decode-full plan variant is still worse than main under emulation (stages
8/10: 275/293; block_n 32: 318-324; warps 2 + block_n 32: 285; splits 48/64: 342/292; hpp 1: 427-449). So session
12's "deeper pipelines might help on the H200" is dead without spending a check.

### Dead ends (reverted; code on branch exp/tdec)

- Transposed decode kernel `_attn_dec` (plan tdec): per head S^T = K Q^T [N, 8] and acc^T = V^T P^T [64, 8]
  (mma N = 8, no padding rows and no off-head blocks; half the MMAs and accumulator registers of the hpp=2
  block-diagonal form). Loading V as [D, N] directly (not tl.trans after the load) matters: 168 -> 142 regs, 349 ->
  261 us. Best H100 decode 32 full 256.5 (block_n 32, stages 3) vs 243.5; emulated 273 vs 260. Dead.
- FA4 lazy rescale in one-warp decode programs (warp-uniform branch, threshold 2^8): H100 tie (243.6 vs 243.5,
  130.4 vs 130.5), emulated tie at equal waves (262 vs 262); it costs 202 vs 148 registers. Dead.
- Gluon verify kernel at sliding length (proxy: full layer, ctx 2048, q_len 8, no RoPE): batch 32 21.0 vs Triton's
  22.1 us, batch 16 14.5 vs 12.9. Not worth porting RoPE + window masks (~0.2% overall).

### Load arithmetic (from check 2363a049, H200)

Per load (ours / SGLang): decode 1 537/1427, decode 16 2065/4468, decode 32 3631/7596, verify 1 593/1597, verify 16
2680/4648, verify 32 4707/7827. Per us saved: batch-1 sliding ~1.3-1.4% overall per us (Codex's native lane), batch-1
full ~0.5%/us, verify full ~0.1%/us. If verify full ran at decode-full speed on the H200 (gap 32/55 us), overall would
be ~2.27x. Verify full's compute (H100 compute-only 106 us at verify 16) is already ~FA3-hdim64 efficiency (42 GFLOP in
106 us = ~400 TFLOPS fp16-equivalent); MUFU alone is ~40 us of floor there.

### Next ideas

1. Merge tail on verify full is still ~5 us (H100) / ~10 us (H200 profile: _gw_verify ends ~124, busy 133.7). An
   in-kernel merge (per-pair arrival counter, the S CTAs of a pair each merge a slice of rows from L2 after all
   arrive) only works for one-wave plans (co-residency) and needs the _new_tokens slot (PDL wait). Maybe 1%.
2. Decode full on the H200: the emulator says wide gathers (BN 64 x 2 warps) keep HBM busier per SM; a decode-full
   kernel with 64-key tiles that stays cheap per tile (unlike block_n 32-64 in the current kernel, which loses on
   compute) would be the thing to build. Codex's lane if it needs hand-written code.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 3/8 checks, 1/3 scored. Account at 04:00: 8/10 checks, 4/10 scored.

## Session 14 (overnight agent, 2026-10-24 04:20-05:20 UTC)

No change to main (kernel = 19b8ade). 0 checks, 0 scored. H100 bench at start 1.982 (decode 16/32 sliding
10.53/15.84, verify 16/32 sliding 12.91/20.24, verify 16/32 full 137.5/258.4).

### Sliding layers at batch 16/32: where the time over the gather floor goes (H100)

- Pure random-row gather at sliding size under a graph, cold L2, best config (/tmp-style script: tools/gather.py
  kernel with N = batch x 2048 rows): batch 32 9.7 us, batch 16 6.0 us. So decode 32 sliding is ~6 us over the
  floor, verify 32 ~10.5, verify 16 ~7, decode 16 ~4.5.
- Ablations (us; decode 16 / decode 32 / verify 16 / verify 32): base 10.7/15.7/12.7/20.4; no merge launch
  9.2/14.7/11.0/17.9; zero KV tiles 4.6/4.9/7.2/9.7; zero tiles + no merge 2.6/3.0/5.2/6.7, and on top of that
  no partial writes 2.5/2.9/4.1/4.5 (verify 32's fp32 partials are 8.4 MB, same order as its 16.8 MB of K/V),
  no `_new` -/-/3.9/5.3, no q RoPE 2.6/2.8/3.9/5.4.
- nsys, verify 32 sliding: _attn_split ~18 us (grid 8x32x2), merge tail after it ~2.4 us.

### Dead ends

- Two independent softmax chains per program over the two halves of its split, merged in registers (plan `ch2`,
  halves the partials at equal parallelism): verify 32 sliding splits 4 20.2-20.8 vs 19.9-20.4; splits 8 26.7;
  verify 16 14.3+ vs 13.0; decode 32 16.9-17.6 vs 16.1-17.4; decode 16 11.7+ vs 10.7. Code not kept.
- FP8 P@V in the Gluon verify-full kernel (branch exp/pv8, plan `pv8`): P rounded to E4M3 (FA3-style; torch emulation
  adds <= 0.04 rel-L2 even on peaked rows, and 2 forced-Gluon cases pass at 0.033/0.086 vs bounds 0.082/0.150), the
  producer stages V^T per head as K-major FP8 instead of converting V to FP16. The transposed smem store is the
  problem: best layout 149 us vs 137.5 (others 151-851; NVMMASharedLayout(transposed=True) 191-241). A wrong-result
  ablation without the transpose is 136.5, so even a free transpose saves only ~1 us on H100. Not worth more unless
  the H200's compute-bound verify (240 vs ~185 decode-equivalent) is shown to gain much more.

### Next ideas

1. Verify 32 sliding's fixed costs (partials 2.2 us + merge 2.4 us + `_new` 1.4 + RoPE 1.2 at zero tiles) are the
   biggest sliding lever left; an in-CTA multi-warp split merge (warps own key ranges, merge through smem, one
   partial per CTA) needs Gluon (warp_specialize partitions with different ranges), not Triton.
2. As before: decode full on the H200 (wide-tile gathers) and the merge tail on verify full.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 3/8 checks, 1/3 scored. Account at 04:30: 8/10 checks, 4/10 scored.

## Session 15 (overnight agent, 2026-10-24 04:33-~05:00 UTC)

### Change (main 32b1762): BF16 split-partial accumulators where they win

Plan key `pdt` ("f32" | "bf16" | "f16", default f32) picks pacc's dtype; stores cast to the pointer's dtype and every
merge (_combine, _combine_hb, fused) loads `.to(tl.float32)`. pml (max, sum) stays FP32. Accumulators stay
UNNORMALIZED, so FP16 is unsafe (a chunk of 640 keys x V up to 448 can pass 65504); BF16 has the range.
H100 paired sweeps, f32 -> bf16: decode 16/32 full 130.6->129.6 / 244.1->242.5 us, decode 1 full 18.2->18.0,
verify 1 full 19.3->18.65, verify 16/32 sliding 13.0->12.5 / 20.5->20.1, decode 16/32 sliding tie. Kept FP32:
decode 1 sliding (+0.2-0.3), verify 1 sliding (+0.05), decode sliding generally, and the Gluon verify-full path
(+1.5 us with bf16 or f16; not understood, maybe the MMA-layout store). Bench 1.968 (main, same session) -> 1.980 /
1.982. Tests 16/16 (errors identical to 4 digits), gluon_cases 8/8. Not checked on the H200 (expected ~+0.5%,
inside the judge's CI; account had 2 checks left today).

### Dead end

- Normalized FP16 partials (store acc / l, merge weights exp2(m - m_max) * l): same byte savings, but the divide
  costs on the critical path: decode 1 sliding +0.5 us, Gluon verify 16 full +1.5. BF16 unnormalized is better.

### Next ideas

1. Why BF16 partial stores cost the Gluon verify-full kernel 1.5 us (verify 32 full writes 33 MB of fp32 partials?
   no: 32 x 4 x 17 x 64 x 64 x 4 = 9 MB). Converting acc to a coalesced BlockedLayout before the store might fix it
   and save ~1 us of merge bytes.
2. As before: verify 32 sliding fixed costs; decode full on the H200 (wide-tile gathers); verify-full merge tail.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 3/8 checks, 1/3 scored. Account at 04:35: 8/10 checks, 4/10 scored.

## Session 16 (overnight agent, 2026-10-24 04:40-~05:30 UTC)

### Change (main 06b7755): Gluon verify-full consumer, thread-local row sums + no duplicate barriers

1. Plan key `lsum` (on for the Gluon path): `_gw_consumer_cv` keeps row sums per thread as `l8 = sum(reshape(p,
   [M, BN/8, 8]), 1)`. In the wgmma accumulator layout column bits 3+ are registers, so that sum stays in
   registers (no quad shuffles); `l8 *= alpha` each tile, and one quad reduction at the end (FA3's deferred
   row_sum). No spills (80 regs, 0 spills; tools/s12/spills.py from branch exp/gluon-pipe).
2. Dropped the explicit `gl.barrier()` before `mbarrier.arrive(empty)` (consumer) and after the producer's
   `fence_async_shared()`: Gluon's arrive already emits `bar.sync <partition>, 128` right before its elected
   arrive (checked in the PTX), so each was a second partition barrier per tile.

H100, paired: verify 16 full 137.1 -> 136.6 us, verify 32 full 257.6 -> 256.8; **compute-only** (cp.async removed,
the bound the H200 hits on verify full) 103.8 -> 101.0 (lsum) -> 100.2 (barriers). Bench 2.005 (other shapes noise).
16/16 tests, gluon_cases 8/8.

### Dead ends

- Gluon store through a coalesced BlockedLayout (`gl.convert_layout` before the partial store), f32 or bf16:
  verify 16 full 158+ us vs 137 (the convert's scratch smem presumably costs the 2nd CTA/SM). BF16 partials on the
  Gluon path are now +1.2 us (was +1.5). Not kept.
- Gluon path forced onto verify 1 full (splits 96/132/200/264): 20.2/20.2/21.7/28.1 vs Triton 18.7. Dead.
- Not tried: the same lsum trick in Triton `_tile` (verify sliding); needs a 2-D l_i through `_new`/`_tile`/`_finish`
  and Triton's reshape may insert layout converts. Each split there is only 4-8 tiles, so maybe a few 0.1 us.
- Register split re-sweep after lsum (cregs/pregs 96/40, 100/36, 104/32): flat at 136.9-137.1 us.
- `gl.exp2` already lowers to `ex2.approx.ftz.f32` (one MUFU, no denormal fixup), so nothing to gain there.

### Judge (UTC day 2026-10-24)

| job | code | speedup (95% CI) | notes |
|---|---|---|---|
| 22381fd8435f4fce (check) | 06b7755 + an unused `CVT` plan flag (default off; my uncommitted experiment rode along through `git checkout main`, same kernels at runtime) | **2.136x** (2.104-2.168) | 16/16; H200 verify 16 full 129.65 -> 125.34 us, verify 32 full 240.27 -> 234.66 (vs check 2363a049 of 824e52a). Noise elsewhere: verify 1 sliding 6.90 -> 7.86 (that path is unchanged), decode 1 full 15.3 -> 16.7, verify 1 full 17.2 -> 16.0 |

The verify-full gain on the H200 (-3.3% / -2.3%) is about what the H100 compute-only ablation predicted, so on the H200
verify full really is bound by consumer compute: keep using the no-gather ablation (cp.async skipped in the producer)
as the metric for verify-full compute work.
| beb63898dbb64991 (submit) | exactly the file checked as 22381fd8 | **2.138x** (2.106-2.170) | 16/16; new account best (was 2.113x d0f30dc0), board rank 20 |

### Next ideas

1. Verify full on the H200 is consumer-compute-bound: keep cutting per-tile consumer work and measure it with the
   no-gather ablation. Left: the row-max quad reduction (2 shfl + 2 FMNMX per row, on the critical path), the acc
   rescale (32 FMUL/thread/tile; FA4's conditional rescale lost in session 12 with a WG-wide vote, though a
   warp-level vote would be enough because a row's quad sits inside one warp), and the ready-wait retries.
2. The lsum trick in Triton `_tile` (verify sliding batch 16/32), if Triton's reshape stays in registers.
3. As before: the merge tail on verify full (~6 us on H100), and decode full on the H200 (wide-tile gathers).

### Budget (UTC day 2026-10-24)

This session: 1 check (22381fd8), 1 scored (beb63898). Claude day: 4/8 checks, 2/3 scored. Account after this session:
9/10 checks, 5/10 scored.

## Session 17 (overnight agent, 2026-10-24 05:31-~06:00 UTC)

### Change (main 3bb3a64): verify-full merge inside the Gluon grid (plan key `gfuse`, on for one-wave Gluon plans)

After `warp_specialize` joins, each `_gw_verify` program arrives on a per-(request, KV head pair) counter word
(arrivals in the low 16 bits, a generation above; the last arrival clears the arrivals and bumps the generation,
the others spin on it), then merges its slice of the pair's 128 rows (RS = cdiv(128, S) rows, all S + 1 partials,
loads unrolled; gate logits prefetched before the wait), applies RMSNorm * sigmoid(gate) and stores the output.
No `_combine_hb` launch. Fences as cooperative groups' grid sync: partition barrier, acq_rel arrival atomic,
relaxed generation bump, `fence.acq_rel.gpu` after the wait, `.cg` loads. Counters persist across calls in
`_counters(pairs, device, "g")` (arrivals end at 0 every call). It only runs when splits * pairs <= 2 * SMs
(one wave); the S programs of a request are consecutive in x, so the waits need only S resident programs.

H100 paired vs main: verify 16 full 136.9 -> 136.0 us, verify 32 full 257.2 -> 255.7 (two rounds each). Tests
16/16, gluon_cases 8/8 (errors identical), alternating verify 32/16/12 calls + 50 graph replays bit-identical.
Bench 1.967 (noisy run, other agent on the GPU: decode 32 sliding 17.2). **Not checked on the H200**: the account had
1 check left today and Codex is keeping it. Expected small (~0.5% on verify full).

How it got there (verify 16 full, H100; base 136.8, no merge at all (wrong output) 132.6):
- Last-arriving consumer WG merges its whole head, inside the consumer partition: 257 us with static unroll (1290
  spills), 163 with a runtime loop (one L2 round trip per slot), and even with the merge never taken 139: the
  extra pointers live through the consumer loop cost spills. Merging after `warp_specialize` avoids that.
- Slice merge after the join: sync only (no merge) 135.0, + merge 137.2, minus the all-thread fence before the
  arrival 136.7, packed counter word (one atomic instead of add + xchg + gen add) 136.3, gate prefetch 136.2,
  relaxed generation bump 136.0. So the sync chain is still ~2 us and the merge ~1 us of the 3.4 us over the floor.
- BF16 partials on top (pdt bf16): tie (136.0-136.2 vs 136.0-136.1; 255.1-256.1 vs 255.7-256.5). Kept FP32.

### Dead end: the same in-grid slice merge on Triton sliding layers (branch exp/sfuse, plan `sfuse`, off)

H100, base -> sfuse: decode 1 4.8 -> 6.3, decode 16 10.5 -> 13.8, decode 32 16.1 -> 18.9, verify 1 6.0 -> 7.9,
verify 16 12.6 -> 15.0, verify 32 20.0 -> 23.5 us. Release-only arrival without the post-wait fence: decode 1 6.05,
verify 32 22.4. The PDL-launched merge (programs already resident, released by griddepcontrol) is much cheaper than
an atomic + spin + fence chain at these sizes. Don't retry for sliding/decode Triton paths.

### Next ideas

1. Verify full on the H200 is consumer-compute-bound (session 16); the in-grid merge may matter more there than on
   the H100 (the H200 profile showed a ~10 us merge tail). Confirm with a check when the budget allows (next UTC
   day), together with whatever else lands.
2. The remaining ~2 us of sync in gfuse: the arrival atomic waits for the partial stores (release); a program
   could keep its own slice rows in registers instead of re-reading, or the merge could skip the own split's
   store for those rows. Small.
3. As before: lsum in Triton `_tile` (verify sliding), decode full on the H200 (wide-tile gathers).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 4/8 checks, 2/3 scored. Account at 05:35: 9/10 checks, 5/10 scored.

## Session 18 (overnight agent, 2026-10-24 05:47-06:15 UTC)

Main 8c4394f (two plan changes below). 0 checks, 0 scored (account had 1 check left today, kept for Codex).

### Changes (plan only; both hold under tools/emu.py H200 emulation)

1. 9a43323 decode 1 full (batch * hkv <= 2, q_len 1): hpp 2 (both KV heads per program, whole 128-byte rows),
   block_n 64, 5 stages (was hpp 1, block_n 128, 6 stages). H100 paired (9 rounds) 18.00 -> 17.38 us; emulated
   19.65 -> 18.69. Verify 1 full: no variant wins (dual, block_n 64, splits 96-256: tie or worse).
2. 8c4394f decode sliding with batch * hkv > 32: hpp 2, block_n 64, 4 warps, 5 stages (was hpp 1, block_n 32,
   2 warps). H100 paired decode 32 sliding 15.97 -> 15.41, B24 -0.9, B48 -1.8, B64 -3.7 us; emulated 17.15 ->
   16.54. hpp 2 at batch 2-16 decode sliding loses (+0.5..+1.2 at B2-8), so the heads <= 32 branch keeps hpp 1.

Bench 1.977 -> 1.986 (decode 1 full 17.1, decode 32 sliding 15.5). 16/16 tests, tools/s18/d1cases.py 6/6.
Expected on the H200: ~+0.3-0.5% overall. Not checked yet.
H100 bench at start 1.977 (verify 16/32 full 136.4/256.6, decode 1 full 17.7).

### Dead ends (H100; base -> variant)

- Warp-level FA4 conditional rescale in `_gw_consumer_cv` (branch exp/wcr, plan `wcr`, off): per warp, keep the old
  row maxima unless one of the warp's rows grew by > 8 (log2), via `vote.sync.any` in inline asm (pack 2: a thread
  holds 2 rows of the wgmma layout, and both rows' flags must go into one vote, else wrong results), and the 32
  acc FMULs behind a `bra.uni` in inline asm (tied "0".."15" constraints, `${:uid}` labels). Correct (gluon_cases
  errors identical), 80 regs, 0 spills, but verify 16/32 full 136.2/256.0 -> 136.3/256.4, compute-only (no gathers)
  101.1/196.9 -> 102.0/197.5. Confirms session 12: the consumer is latency-bound, not issue-bound; removing ALU ops
  doesn't help, the vote adds latency. Don't retry rescale-skipping.
- Gluon verify full BN 32 (2 CTAs/SM): 150.7 (compute-only 121.6) vs 136.2 (101.1). So FA3 pipelining at BN 32
  (which would fit in the registers) would have to win back 20% first. Not tried.
- Gluon verify full at 1 CTA/SM (splits 8, maxnreg 152, cregs 184): BN 128 141.5 (compute-only 115.5), BN 64 154.0
  (130.9). BN 128 is much better than BN 64 there, but still behind 2 CTAs/SM BN 64.
- Sliding layers with fewer splits (verify 32 sliding: splits 4/2/1 with block_n 32-128: 20.6-32+ vs 19.6; decode 32
  sliding splits 4/2/1: 18-35 vs 15.9). A single CTA's gather throughput is low (splits 2 = ~5 GB/s per SM), so the
  sliding kernel needs the many-program split structure; dropping the merge that way doesn't pay.
- Decode 1 full plan re-sweep (splits 132/192/264, block_n 64 with 6-11 stages, hpp 2, 8 warps): 17.65-18.85 vs
  18.2 (same session); splits 132 at 17.65 is noise-level. Floor for that shape (tools/s18/gather1.py: 80000 random
  128-byte K+V rows, graph replay, cold L2): 11.5 us on H100 at any chunk 160-640. So decode 1 full is ~6 us over
  its gather floor (q prologue, split 0's new token, partial writes, the 8-program merge); that's ~0.5%/us overall,
  the largest per-us lever left after batch-1 sliding.

### Tools added (main)

- tools/s12/{mkabl,runabl,one,spills}.py (copied from exp/gluon-pipe): `python tools/s12/mkabl.py` writes the
  no-gather ablation to /tmp/s12/abl (mkdir it first); `LOADS="verify 16,verify 32" python tools/s12/runabl.py
  <dir> '<json list of plan overrides>'` times verify full for either copy.
- tools/s18/cases.py '<json>': gluon_cases.py with plan overrides on the Gluon path. tools/s18/ptx.py '<json>' out.ptx:
  regs/spills + PTX of the verify-16 Gluon kernel. tools/s18/gather1.py: batch-1 full-layer gather floor.

### Batch-1 full breakdown (H100, main before change 1; tools/s18/mkabl1.py ablations, wrong results on purpose)

decode 1 / verify 1 full: all 17.6 / 18.6; no merge launch 15.6 / 15.6; no KV tiles 3.4 / 4.3; no tiles + no merge
1.5 / 1.8; no `_new` 17.2 / 18.8. So the merge costs ~2 us (decode) / ~3 us (verify) and the tiles ~14 us vs an
11.5 us pure-gather floor. Merge tweaks that didn't help: all loads of the one-pass `_combine` issued together
(tie), 2 or 8 warps instead of 4 (+0.1..+1.2 us).

### More dead ends

- Verify sliding as dual chains (dual=True, splits 8/16): verify 32 sliding 20.9+ vs 19.4, verify 16 15.4+ vs 12.6.
- Decode 16 full wider tiles (block_n 32/64, warps 2/4, splits 16-64, stages 4/6): best 130.3 vs 129.4 on H100,
  and emulated 152.3 vs 138.5. Dead on both.
- Verify 16 sliding stages 6 (12.30 vs 12.53, one sweep; unpaired, likely noise).

### Next ideas

1. Check main 8c4394f on the H200 next UTC day (gfuse from session 17 + the two plan changes; expected ~+0.5-1%).
2. Batch-1 full merge (2-3 us on H100): Codex's native merge covers it on the codex branch; in Triton the
   remaining option is fusing it into the last split (lost before at batch 1).
2. Gfuse for verify full still unchecked on the H200; check it next UTC day together with anything else.
3. Decode full on the H200 (74% of HBM BW): wide-tile gathers; needs a check to evaluate.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 4/8 checks, 2/3 scored. Account at 05:47: 9/10 checks, 5/10 scored.

## Session 19 (overnight agent, 2026-10-24 06:13-~08:00 UTC)

Main unchanged (8c4394f kernels). 0 checks, 0 scored (account had 1 check left today, kept for Codex).
New work on branch **exp/gdec** (a10906f, pushed): a Gluon decode-full kernel. **Check it on the H200 first thing next
UTC day**; if it wins there, merge exp/gdec into main (it only touches decode full layers at batch * hkv > 2).

### Finding: decode full is compute-bound on the H200, not only gather-bound

tools/s19/cbound.py times each full-layer shape as is and with `page_table % 4096` (every K/V read hits L2, so
what's left is the kernel's own work). H100, main: decode 16/32 full 129.4/241.8 real vs **83.4/157.0 all-L2**;
verify 16/32 full 136.3/256.2 vs 110.3/210.5. The H200 judge has the same SM count and clocks, and its decode 32
full kernel time is 181 us (check beb63898), so the 157 us of kernel work is close to the H200 bound. The pure
gather kernel (tools/s19/gl2.py) is 79-83 us all-L2 at the same size, so our decode kernel's work is ~2x a plain
gather's. Why: Triton's decode-full tile (1 warp, block_n 16, hpp 2, 149 regs, 12.4 KB smem) uses mma v2 (M=16), and
on sm_90 Triton's MMAv2 path **upcasts every FP8 K byte to FP16** (ttgir: tt.fp_to_fp before tt.dot; SASS 302 F2FP,
256 LDS (V read a byte at a time into the dot-operand layout), 88 HMMA); K/V ring is only 2 tiles of 16 rows.

### exp/gdec: `_gd_full` (plan key `gdec`, module flag `USE_GDEC`)

Gluon, one warp per program, same grid as Triton's (splits x batch x KV-head pairs, 1024 programs), block-diagonal
[16, 128] queries as before, but:
- QK^T as **native FP8 `mma.sync m16n8k32 e4m3`** via Gluon's `ampere.mma_v2` with fp8 DotOperandLayouts (k_width 4;
  works on sm_90, exact): no K conversion.
- V read from smem 16 bytes per thread (BlockedLayout [1,16]), converted to FP16, `convert_layout` to the B operand.
- Explicit cp.async ring (block_n 32, NS 2, rows past hi zero-filled by the mask, no clamped re-reads), page
  indices one tile ahead, per-request balanced chunks.
- The step's new token: written to the cache and attended (one-key partial into slot S) by the last split's program,
  so no `_new_tokens` launch (that launch + its PDL chain cost ~5-8 us). Merge: `_combine` over S + 1 slots, with
  splits = 2048 // heads - 1 so S + 1 is a power of two; BF16 partials.
- Gluon pitfall: `gl.convert_layout` of an FP8 [16, 128] tensor from BlockedLayout([1, 4], [2, 16]) into the FP8 A
  DotOperandLayout gave wrong values (zeros became 384.0); from BlockedLayout([1, 16], [4, 8]) it is right.

Results (all paired, same session):
- All-L2 work, H100: decode 16/32 full 84/157 -> 69/132 us.
- **H200 emulation** (tools/emu.py, 40 SMs parked, splits scaled 92/132 for both; tools/s19/ab.py): B8/16/32/64
  77.8/137.3/261.2/504.5 -> 75.6/131.6/250.2/486.0 us (-2.7/-4.1/-4.2/-3.7%).
- **H100 real**: B8/16/32/64 74.1/129.2/241.8/464.4 -> 74.4/131.3/243.9/469.8 (+0.4..+1.5%). So the H100 bench will
  show it as a small loss (~-0.2% bench geomean); only the judge can say. Expected on the H200 if the emulation
  holds: decode 16/32 full ~-4 / -9 us, ~+1.2% overall.
- Tests 16/16; tools/s19/cases.py (9 decode-full shapes through `kernel_fn`, batch 2-64, page 1/64, kv_scale,
  edges, ctx 100-80K) 9/9 pass.
- Dead variants: block_n 16 (NS 4-5): all-L2 time up (99-196 us); block_n 64 (NS 2): 160-301 us real; splits 31/47
  at decode 16 or 23/47 at decode 32: worse; page indices two tiles ahead: no change; V straight from fp8 smem into
  the dot-operand layout (byte loads, as Triton does): all-L2 +2-3 us.

(tools/s19/cbound.py and gl2.py are on main too; gdec.py (standalone prototype), cases.py and ab.py are on exp/gdec.)

### Later in the session (exp/gdec 2nd commit)

- Plan key `tpv` (off): transposed P@V, O^T [D, G] += V^T [D, BN] @ P^T [BN, G] per head (only the diagonal blocks;
  P^T from the S registers via reshape/permute/split + convert_layout, V staged to an FP16 smem tile, 8 KB more smem).
  Correct (cases 9/9). H100 real decode 16/32 full 131.2/244.5 -> 129.5/242.9 (a tie with Triton), but all-L2 work
  69/132 -> 78/145 and emulated H200 a tie (131.9/249.5 vs 131.7/250.7). If the H200 check of gdec shows it bandwidth-
  rather than work-bound, try tpv; otherwise drop it.
- `_gd_full` compiles to 242 registers at NS 2 + BF16 partials (168 in the f32/NS 3 prototype); with 1024 programs
  that's still one wave (8 warps/SM), so it doesn't matter unless splits grow.
- Bench on exp/gdec: 1.998 (main at session start 1.991; decode 16/32 full 129.8/243.0 vs 129.3/242.1).

### Next ideas

1. Check exp/gdec on the H200 (compare decode 16/32 full against check beb63898: 98.46 / 185.10 us). If it wins,
   merge to main and tell Codex (it changes `_plan` for decode full at batch * hkv > 2; Codex's native batch-1 path is
   B * Hkv <= 4, so batch 2 decode full moves from Triton to gdec on main).
2. Make `_gd_full` win on the H100 too: 168 registers / 17 KB smem per program. A transposed P@V (O^T = V^T P^T with
   only the two diagonal 64 x 8 blocks: half the PV MMAs and a 32-register accumulator instead of 64, P^T straight from
   the S accumulator registers) would cut registers and per-tile work further.
3. The same FP8-QK^T Gluon tile for **decode sliding** at batch 16/32 (Triton's sliding decode path also runs mma v2
   with upcast K), and check whether verify sliding (M = 64, wgmma) has a similar hidden conversion.
4. Verify full: all-L2 time is 210 us vs 233 on the H200, so it's even more compute-bound than decode full there.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 4/8 checks, 2/3 scored. Account at 06:15: 9/10 checks, 5/10 scored.

## Session 20 (overnight agent, 2026-10-24 06:40-07:25 UTC)

**Main 8aefd49 is a CANDIDATE, not yet checked as a whole: check it first thing next UTC day** (expected ~2.19x vs board
2.138x; if it holds, submit it). It is 06b7755's plans (the code of scored beb63898) + Gluon decode full (`_gd_full`,
from exp/gdec, batch * hkv > 2) + session 17's gfuse (verify full), with session 18's two plan changes reverted and
the new Gluon sliding-decode path off (`USE_GSLIDE = False`). Every scored shape's code path was timed on the H200 in
either beb63898 or f3e099c1; recombining their per-shape medians gives 2.197x (the same arithmetic reproduces
f3e099c1's own 2.1448x exactly).

Branch **exp/gslide** (from exp/gdec, so it also carries session 19's Gluon decode-full kernel): batched sliding decode
on the same Gluon one-warp kernel `_gd_full`. Main unchanged (8c4394f kernels) until the H200 check says otherwise.

### Why

tools/s19/cbound.py on sliding layers (H100, real vs all-reads-in-L2): decode 16/32 sliding 10.5/15.3 vs 7.9/11.5 us,
verify 16/32 12.5/19.7 vs 10.6/16.6. So sliding layers are mostly kernel work too, and decode sliding runs Triton's
MMAv2 path (FP8 K upcast to FP16, as session 19 found for decode full). Plan re-sweeps of the Triton path (splits
16/32, block_n 32/64, warps 2/4, stages 3-5) were flat or worse.

### Change (b95d710)

- `_gd_full` gained `WINDOW` (keys [max(seq - 2047, 0), seq) in S tile-aligned chunks), `ROPE` (q and the new K
  roped from the [2^21, 64] cos|sin table, partner element via `c ^ 1`), and `SEP`: one extra program per (request,
  KV pair) (split index S) handles only the new token and writes slot S; key splits store only if split < S.
  Without SEP the last key split also does the new token and becomes the critical path: S=16 lost to S=15 (whose
  last split happens to have no keys) until SEP.
- `_decode_gluon` merges with `_combine_hb` (hb 2, one pass, 1 warp) when the plan has hb.
- `_plan`: decode sliding with pairs = batch * hkv / 2 >= 12 (G 8, D 64): gdec, sep, block_n 32, NS 2, BF16
  partials, splits 32 if pairs * 32 <= 800 else 16 if pairs * 16 <= 800 else 8.
- tools/s20/cases.py (11 sliding cases through kernel_fn, plan override json), near.py (paired A/B vs Triton per
  batch, splits rule SP/pairs, optional emulation), ab.py (paired USE_GDEC A/B per scored decode shape).

### Measured (H100 unless noted; paired)

- Decode 16/32 sliding 10.56/15.34 -> 9.62/14.06 us; emulated H200 (92 SMs) 11.54/16.37 -> 10.26/14.48.
- Per batch (H100, best splits): B12 -0.6, B16 -1.0, B24 -1.9, B32 -1.5, B48 -2.6, B64 -4.5 us; B1-8 lose
  +0.2..+1.8 (so they stay on Triton).
- Bench 1.978 (main at session start) -> 1.993. Tests 16/16, tools/s20/cases.py 11/11, tools/s19/cases.py 9/9,
  tools/nearby.py 48/48 pass, no cliffs.
- Losing variants: block_n 64 (S 15: 14.2 vs 9.7 at B16), NS 3 (+1-1.5 us), block_n 16 NS 3, S 64 at B<=16 (1-tile
  programs), FP32 vs BF16 partials a tie.
- Note: tools/s20/ab.py doesn't rescale gdec decode-full splits for 92 SMs, so under emulation decode full shows
  +24/+50 us (1024 programs x 242 regs = two waves on 92 SMs). Artifact; session 19's tools/s19/ab.py rescales.

### Judge

- check f3e099c1c36145dc on b95d710 (the account's last check today): **2.145x [2.113, 2.177]**, 16/16. Per shape
  (H200 us, beb63898 -> f3e099c1): decode 1 sliding 6.47 -> 6.35, **decode 1 full 15.21 -> 17.64** (session 18's
  hpp 2 plan: H100 win, H200 loss), **decode 16 sliding 10.36 -> 13.56, decode 32 sliding 14.56 -> 15.37** (Gluon
  sliding: H100 and emulation said -1/-1.3 and -1.3/-1.9; the judge says a loss), **decode 16 full 98.46 -> 92.63,
  decode 32 full 185.10 -> 173.84** (Gluon decode full: wins as emulated), verify 1 sliding 7.76 -> 7.42, verify 1 full
  16.53 -> 16.41, verify 16 sliding 12.73 -> 12.63, verify 16 full 125.30 -> 124.65, verify 32 sliding 18.64 -> 18.34,
  verify 32 full 232.93 -> 231.75.
- Judge profile for decode 16 sliding (kernel times, warm): `_gd_full` 6.6 + `_combine_hb` 2.3 = 8.9 us vs Triton
  7.1 + 1.7 = 8.8 us, a tie; but cold-L2 graph replays median 17.9 vs 14.5 us raw. So the loss is in the cold
  start (not visible on the H100's bench or emulation), and the merge is slower too: S + 1 = 33 slots pads
  `_combine_hb` to S_PAD 64. Untested guesses: the Gluon kernel's cold instruction fetch / constant loads (one-warp
  programs, 528 of them), the merge padding. Don't retry sliding on _gd_full without an H200 check in the loop.
- submit 10d12495c672420b: the exact file of f3e099c1 (b95d710; it beat board 2.138x): **2.1605x [2.128, 2.200]**,
  16/16, new account best, rank 21. (Profiles of it land in /tmp/s20sub/profiles, not the repo.)
- Lesson: two H100-and-emulation wins (decode 1 full hpp 2, Gluon sliding) were H200 losses, while the H200 judged
  the Gluon decode full better than the H100 did. Small latency-bound shapes don't transfer; budget checks for them.

  H200 reference (check beb63898, main kernels then): decode 16/32 sliding 10.36/14.56, decode 16/32 full
  98.46/185.10, decode 1 full 15.21 us.

### Also measured

- `sep` on decode full gdec: tie/worse (130.1-130.7 vs 130.6 decode 16, 244.5-245.3 vs 243.1 decode 32).
- Verify sliding (Triton, M 64) already runs QK^T as native FP8 wgmma (ttgir: warp_group_dot on f8E4M3 operands;
  only V converts to FP16), so the session 19 "hidden K upcast" doesn't exist there.

### Next ideas

1. Check main 8aefd49 on the H200 (new UTC day); submit if it beats the board (now 2.1605x from 10d12495).
2. Decode 32 sliding on the H200: session 18's hpp 2 plan (H100 15.97 -> 15.41) is unchecked there; try it in a
   later check together with something else.
3. Gluon decode full: tune on the H200 only (the H100 can't see its win). E.g. splits so S + 1 = 64 vs 32, NS 3.
4. Verify full is the largest remaining time (124.7/231.8 us on the H200 vs SGLang 205/389).

### Budget (UTC day 2026-10-24)

This session: 1 check (f3e099c1), 1 scored (10d12495). Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks,
6/10 scored.
