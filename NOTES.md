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

## Session 21 (overnight agent, 2026-10-24 07:31-08:00 UTC)

**Main 779bb6d = session 20's candidate 8aefd49 + balanced splits for verify full (`USE_GBAL`). Still UNCHECKED on the
H200. Check it first thing next UTC day** (this problem's account limit is 10 checks/day, not 20, and all 10 were
used today; Claude 3/3 scored used). In that check, compare verify 16/32 full against f3e099c1 (124.65 / 231.75 us);
every other shape against beb63898 / f3e099c1 as in session 20. If verify 16 full gets worse but verify 32 better,
gate gbal on pairs >= 24 or so; if both get worse, set `USE_GBAL = False` (the rest of the candidate is unaffected).

### Finding: request-length spread costs the full layers when compute-bound

Agentic lengths are uniform in ctx +- 7.5%, and every plan cut each request into the same number of splits, so the
longest request's programs set the time. tools/s21/imbal.py (time with all lengths cut to the shortest, scaled by
mean/min): H100 real decode 16/32 full, verify 16/32 full: no loss except verify 32 (-2.3%) because the H100 is
bandwidth-bound; all-L2 (page_table % 4096, compute-bound like the H200): decode 16 full 70.9 vs 68.3 balanced, decode
32 133.9 vs 133.5, verify 16 109.7 vs 106.9, verify 32 208.7 vs 200.5.

### Change: plan key `gbal` (Gluon verify full, pairs >= 16)

- Each KV-head-pair index gets P = 2 * SMs // (hkv / 2) programs (264 on 132 SMs: all slots, not splits * batch).
  Program p takes tiles [p*T//P, (p+1)*T//P) of all requests' tiles laid end to end (T = sum of cdiv(seq, 64)), so a
  range can span requests: one segment per request, partial stored at id p + b (unique), new tokens at P + B + b
  (`_new_tokens` NID/NBASE), partials [HKV, NID, 64, 64]. `_gw_bal` / `_gw_seg` compute the ranges (vector of the
  batch's lengths, associative_scan), `_gw_producer_bal` / `_gw_consumer_bal` loop over segments (ring position
  carried; the consumer loads/norms each segment's queries into its q smem). Merge: `_combine_bal` (Triton, PDL;
  finds request b's programs from closed forms + a predicate, 2 heads per one-warp program). No in-grid merge (gfuse).
- Pitfall: the lengths vector's layout must sit inside one warp (size_per_thread B_PAD // 32): with B_PAD 64 spread
  over 2 warps the scans/sums went through shared memory and cost +28 us (all-L2 batch 32 forced to B_PAD 64).
- H100 paired (tools/s21/ab.py, emuab.py, near.py): verify 16/32 full all-L2 109.2/212.8 -> 102.3/196.4 us, real
  136.0/255.6 -> 137.6/251.4, emulated H200 (92 SMs; partly an artifact: the old plan fills only 160 of 184 slots
  there) 161.1/313.6 -> 154.7/291.5. Nearby real: batch 6/8/12/16/20 +2.0/+3.8..4.6/+0.3/+2.4/+1.9 (the lost gfuse
  merge; hence pairs >= 16), 24 (ctx 120K)/32/40/48/64 -4.6/-4.8/-8.8/-7.1/-8.4.
- Interpolating the H200 between H100 real and all-L2 (it sits ~45-55% of the way, from f3e099c1): expected verify
  16 full ~-2 us, verify 32 ~-10 us: about +0.7% overall. Tests 16/16, gluon_cases 8/8, forced-gbal cases 8/8.

### Next ideas

1. Check main 779bb6d (see top of this section); submit if it beats the board (account best 2.1605x).
2. (Tried, dead: see below) in-grid merge for gbal.
3. The same balancing for decode full (`_gd_full`, all-L2 decode 16 full 70.9 -> ~68.3): needs segments in the
   one-warp kernel; ~0.3% overall at best.
4. Two-segment programs pay a q reload (dependent on the lengths load); the consumer could prefetch the second
   segment's queries (no smem left for a second q buffer at 2 CTAs/SM: 106.8 KB each).

### Dead end: in-grid merge for gbal (exp/bal, plan keys bfuse/rpad/mu, off)

`_gw_bal_fuse`: with tiles >= 1 per request and pe = min(P, T) working programs (also on exp/bal, not on main),
request b's programs are ((excl_b+1)*pe-1)//T .. (incl_b*pe-1)//T; only a program's first and last segments can be
shared, so it arrives on both counters first (gfuse's generation words), waits, then merges its rank's R_PAD-row
chunks of each segment (online over the partials, U unrolled loads). Correct (cases 8/8) but slower than gbal's
separate `_combine_bal`: B16/32 real +1.6/+2.5 us, all-L2 +2.8/+4.0 (rpad 16 the same, rpad 4 worse, mu 16/32 the
same; regs/spills/smem unchanged). Versus the old gfuse plan at small batch: B8 tie, B12 +2.4. Cause not found.
(Arrive-wait-merge one segment at a time would chain every request's merge behind the previous one's: avoid.)

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored (none left: account 10/10 checks for this problem). Claude day: 5/8 checks, 3/3
scored.

## Session 22 (overnight agent, 2026-10-24 07:55-08:10 UTC)

**Main unchanged: 779bb6d's kernels (session 21's candidate) are still UNCHECKED on the H200. Check main first thing
next UTC day** (see session 21's top paragraph for what to compare). The account had 0 checks left all session.

Experiments on branch **exp/par** (pushed; plan keys `par`, `pv8`, both off, plus an env `PSCALE`). Both
are dead ends at their current form. Tools: tools/s22/spills.py (regs/spills/converts of the Gluon verify kernel for
plan overrides; `LOAD=` picks the shape), tools/s22/ptx.py (dump its PTX). On exp/par only.

### Dead end: 1 CTA/SM with 4 consumer warpgroups (plan par 2)

Idea: 2 consumer WGs per KV head taking alternate ring tiles (own m/l/acc, PAR partial sets merged by
`_combine_bal`), so BN 128 tiles (half the per-key rescale/reduction/barrier work) fit at the same 4 consumer WGs/SM.
Correct (forced gbal cases 8/8), but slow (paired tools/s21/ab.py, all-L2 = page_table % 4096):
- BN 64, NS 6, PW 4: verify 16/32 all-L2 120.9/225.2 vs 103.8/194.3 base; PW 8 (maxnreg 80): 117.4/217.1. So the
  1-CTA layout itself loses ~13% at equal tile size (cause not found; not the producer).
- BN 128, NS 3: 230-264 (all-L2), 300-600 real; 36-46 bytes of spills at maxnreg 96 (20 warps cap it at 96).
- BN 128 can't run at 2 CTAs/SM either: ~64 KB smem per stage (fp8 K, V + fp16 V halves).
- Register fact (corrects a guess I made mid-session): ptxas does allocate past the launch count in setmaxnreg regions
  (base kernel: "Used 80 registers" but SASS uses up to R99), so `cregs` really is the consumer's compile budget.

### Measured ceiling: FP8 P@V on the gbal path (plan pv8)

`pv8 1` = P as E4M3 (`p.to(fp8)` into a k_width-4 dot operand, O_L instr k 32) and the producer stages V^T as FP8
(transposed NVMMASharedLayout) instead of converting V to FP16. Correct at PSCALE 0 (errors 0.030/0.036/0.073/0.051 vs
FP16 P 0.025/0.030/0.072/0.042 on the batch 12/16/32/64 cases). Scaling P by 2^8 (PSCALE 8) breaks results: the
new-token partial from `_new_tokens` is unscaled (store m - 8 to fix, if ever needed).
- `pv8 2` (wrong-result ablation: no transpose, same bytes): verify 16/32 all-L2 103.7/196.2 -> 99.6/186.3 us, real
  137.3/251.1 -> 136.9/250.9. With 4 stages (fits now): no further gain. So a free transpose is worth ~-4/-10 us
  compute-only, maybe -2/-5 us on the H200 (~+0.3% overall).
- `pv8 1` with the simple VT register layout ([4,1] per thread, threads along keys): 788/1549 us (shared-memory bank
  conflicts: 32 threads read the same bank of 32 rows 128 B apart, byte-wide).
- To make it pay, the producer needs FA3's FP8 V transpose (SmemTransposeFp8_64x64: ldmatrix.trans b16 + byte PRMT +
  stmatrix), or a 4x4-byte register transpose with a swizzled v_ring (cp.async to `chunk ^ (row % 8)`) so the
  LDS.32/STS.32 are conflict-free. Gluon may not expose ldmatrix.trans for this; inline asm would be needed.

### Next ideas

1. Check main 779bb6d (session 21's candidate) when checks are available; submit if it beats the board (2.1605x).
2. FP8 P@V with a conflict-free V transpose in the producer (above). ~+0.3% at best; precision margin shrinks a little.
3. Session 21's ideas 3-4 (balanced decode full, second-segment q prefetch) are still open.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored (account 10/10 checks used; Claude day 5/8 checks, 3/3 scored).

## Session 23 (overnight agent, 2026-10-24 08:08-08:30 UTC)

**Main 576ad58 = 779bb6d (session 21's gbal candidate, still unchecked) + `USE_GD1` (decode 1 full on the Gluon
one-warp kernel). Both UNCHECKED on the H200. Check main first thing next UTC day.** They touch different shapes, so
one check evaluates both: verify 16/32 full vs f3e099c1 (124.65 / 231.75 us; see session 21 for the gbal fallback
rule) and **decode 1 full vs beb63898's 15.21 us**. If decode 1 full is worse, set `USE_GD1 = False` (session 18's
batch-1 plan also won on the H100 and the emulator but lost on the H200; gdec won on the H200 at batch 16/32).

### Leverage (why batch 1)

From the H200 per-shape numbers (beb63898/f3e099c1), 1 us is worth about: decode/verify 1 sliding 1.4/1.3% overall,
decode/verify 1 full 0.5/0.44%, batched sliding 0.2-0.4%, verify full ~0.06-0.1%. Getting verify 16/32 full to
decode-full speed on the H200 (93/174 us) would by itself give ~2.37x (the leaders' level).

### Change: `USE_GD1` (plan for batch * hkv == 2, q_len 1)

- `_gd_full` with 511 splits + the new token's slot (512 BF16 partials), block_n 32, 2 stages; `_combine` in one pass
  (new plan keys `csb` 512 = SB, `cw` 8 warps; they only affect `_decode_gluon`'s plain `_combine` path).
- H100 (tools/s23/abl.py = total and no-merge): old Triton plan 17.47 total / 15.30 attention; gdec 511 with the old
  merge (SB 128, 4 warps) 16.6 / 12.3; with csb 512 cw 8 15.7 / 12.4. Paired (tools/s23/ab1.py, 7 rounds) 17.86 ->
  15.90 us, every round -1.6..-2.9. Emulated H200 (tools/emu.py 40) 20.03 -> 15.86 (the emulator overstates the old
  plan: the judge measured 15.2). Tests 16/16, tools/s18/d1cases.py 6/6.
- Sweeps that lost: splits 255/383/447/639/1023 (19.2/17.3/16.1/17.6/24.5), stages 3 (17.1), block_n 16 stages 4
  (18.6), cw 4 / 16 (16.0 / 16.2), csb 256/128 with cw 8 (16.5/17.5), FP32 partials (+0.3), `sep` (new-token program,
  tie/+0.2), `_combine_hb` hb 2 (19.4, or 184 at cw 1).
- `cw` 2/8 for decode 16/32 full's merge: noise.

### Dead ends

- Gluon verify kernel (`_gw_verify`) at batch 1 (verify 1 full): splits 264 no-gfuse 28.1 total / 17.1 attention,
  splits 128 21.0 / 16.5 vs the Triton plan 18.75 / 16.1; gfuse at batch 1 is pathological (56-212 us).

### Next ideas

1. Check main 576ad58 (above). Submit if it beats the board (account best 2.1605x).
2. Decode 1 full's merge is now 3.2 us of 15.7 (512 partials, 16 one-head programs). A 4-warp `_gd_full` variant
   where each warp is its own split (3-D tensors with warps on dim 0; Gluon's mma_v2 goes through semantic.dot, which
   should take batched rank-3 operands) merging its 4 partials on chip would cut that to 128 slots (~2.2 us, as the
   Triton plan's merge). ~-1 us on H100, only worth it if the H200 confirms USE_GD1.
3. Verify 1 full: attention 16.1 us vs the 11.5 us gather floor, merge 2.6. No plan found better (sessions 18/23).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored (account 10/10 checks used). Claude day: 5/8 checks, 3/3 scored.

## Session 24 (overnight agent, 2026-10-24 08:17-09:00 UTC)

**Main de522ae = 576ad58 (session 21's gbal + session 23's USE_GD1, both still UNCHECKED) + one-warp merge programs
for Gluon decode full (`cw` 1 when splits + 1 <= 128). All three UNCHECKED on the H200; they touch different shapes, so
one check evaluates them all: verify 16/32 full vs f3e099c1 (124.65/231.75 us; gbal fallback rule in session 21),
decode 1 full vs beb63898's 15.21 us (USE_GD1), decode 16/32 full vs f3e099c1 (92.63/173.84 us; if worse, drop the
`cw` 1 line in `_plan`'s gdec branch).** The account had 0 checks left all session (10/10 for this problem today).

### Change (de522ae): `cw` 1 for the gdec decode-full merge

- Merge tails (total minus no-merge, tools/s23/abl.py, H100): decode 16/32 full 5.5/4.4 us (default `_combine` at
  4/2 warps), sliding 1.6-3.4 us. The H200 profile of f3e099c1 showed ~6 us after `_gd_full` on decode 16/32 full.
- One-warp `_combine` programs: paired (tools/s21/ab.py, 7 rounds) decode 16 full real -1.76 / all-L2 -0.98, decode
  32 full -1.66 / -1.07. Nearby decode full (tools/s24/near.py '{}' '{"cw":1}'): B8/12/16/24/32/48/64
  -1.7/-1.0/-1.7/-2.0/-2.1/-2.1/-1.0 us, B2/4 (256-512 slots) +1.0/+0.7, hence the 128-slot rule. Tests 16/16, all
  nearby cases PASS. Bench decode 16/32 full 129.6/242.1 -> 128.1/241.4.
- Why (unproven): `_gd_full` uses 242 registers, 8 programs per SM leave 2048 registers, so few PDL merge programs can
  be resident before the attention grid drains. But capping registers doesn't help: `_gd_full` maxnreg 232/224/208
  (temporary plan key `gmr`) tie/noise, merge maxnreg 64/40 +4/+11 us (spills), `csb` 32/16 +2 us, hb 2/4 at cw 1
  tie / +3 at decode 16.

### Measured, no change

- Verify full `_combine_bal`: sb 16/8, hb 1, cw 2 all lose (+1..+2.9 us). Verify 16/32 sliding `_combine_hb`: hb 4/8,
  cw 2 tie or lose. The merges other than decode full's are already at their best settings.
- Window mask on unmasked sliding tiles (only split 0's first tile can need it): dropping it entirely (wrong-result
  upper bound) gives verify 32 sliding -0.45, decode 32 -0.2 us, within noise (verify 16 moved -0.24 though it
  never takes that path). Not worth peeling.
- Per-load arithmetic from f3e099c1 (H200): decode 1/16/32 2.73/2.27/2.20x, verify 1/16/32 2.70/1.79/1.71x. Verify
  16/32 full remains the weak spot (compute-bound consumer; see sessions 12/13/16/22 for everything tried there).
  Board: #1 is at 2.726x, #10 at 2.317x; we're #21 at 2.161x.

### Next ideas

1. Check main de522ae (above) when the UTC day resets; submit if it beats the board (account best 2.1605x).
2. Verify full consumer: still the only large lever (~+2% overall per 30 us at verify 32). Everything cheap in
   Gluon has been tried; FP8 P@V with a conflict-free producer-side V transpose (session 22) is the remaining
   Triton/Gluon-lane idea (~+0.3%).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 25 (overnight agent, 2026-10-24 08:27-09:05 UTC)

**Main unchanged (de522ae kernels). Still UNCHECKED on the H200: check main first thing next UTC day, exactly as
session 24's top paragraph says** (verify 16/32 full vs f3e099c1 for gbal, decode 1 full vs 15.21 us for USE_GD1,
decode 16/32 full vs 92.63/173.84 for cw 1). The account had 0 checks left (10/10 for this problem today).
Baseline reproduced: H100 bench 2.014x (decode 1 sliding 4.86, verify 1 sliding 5.92 us).

### Dead end: in-grid merge via a thread-block cluster (batch-1 sliding)

Idea: batch-1 sliding pays ~1.4 us for the separate PDL merge kernel (session 5: total minus attention-only), at
~1.4% overall per us. Hopper clusters (up to 16 CTAs) could merge the splits in-grid behind a hardware cluster barrier
instead of a grid boundary. Probes in tools/s25/ (no change to submission.py):
- Gluon can launch clusters: `num_ctas=NC` gives a cluster of NC CTAs along grid x (program_id(0) = cluster index);
  every layout then needs `cga_layout` bases (log2 NC of them). The rank comes from
  `inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;")`. Broadcast bases ([0, ...]) mean "replicated": global
  stores from such a layout are done by CTA 0 only (probe.py). A CTA-split leading dim ([1 << i, 0, ...]) gives each
  CTA its own slice (probe2.py). 16-CTA clusters launch fine.
- Reductions over a CTA-split dim compile to DSMEM (`mapa`, `ld.shared::cluster`, many `barrier.cluster`) and are
  correct, but need a monkeypatch: Gluon 3.8 crashes comparing layouts whose cga_layout is a list
  (`'list' object has no attribute 'type'` in core._type_for_tuple_values; probe5.py wraps lists as constexpr).
- Cost (tools/s25/cmerge.py, warm back-to-back launches, 2 clusters, partials [NC, 16, 64] fp32): plain copy
  1.33/1.45 us (NC 8/16); DSMEM reduction merge 6.44/7.97 us (hopeless); exchange through L2 (each CTA stores its
  slice, `cluster.barrier()`, all reload in a broadcast layout and reduce locally) 2.78/3.17 us, i.e. +1.45/+1.7 us
  over the copy: the same as the PDL merge kernel it would replace. On top of that it would need 16 splits of 128 keys
  (one cluster per KV head), which the Triton path already showed is slower than 32 x 64 (session 6). Not worth
  building.
- Also checked: `_gd_full` (Gluon one-warp) at decode 1 sliding is 7.0 us (splits 31) vs Triton's 4.8; the merge
  kernels already preload the gate before `gdc_wait`.

### Next ideas

1. Check main de522ae (session 24's list); submit if it beats the board (account best 2.1605x).
2. Unchanged from session 24: verify full's consumer is the only large lever; FP8 P@V needs a conflict-free V
   transpose (inline PTX: Codex's lane, or hand it over via COORD.md).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 26 (overnight agent, 2026-10-24 08:34-09:00 UTC)

**Main kernels unchanged (de522ae). Still UNCHECKED on the H200: check main first thing next UTC day, exactly as
session 24's top paragraph says** (verify 16/32 full vs f3e099c1 for gbal, decode 1 full vs 15.21 us for USE_GD1,
decode 16/32 full vs 92.63/173.84 for cw 1). The account had 0 checks left (10/10 for this problem today).

### Dead end: thread-local row sums in the Triton kernel (branch exp/tlsum, plan key `tlsum`)

The Gluon verify-full `lsum` trick (session 16) ported to `_tile`/`_new`/`_finish`: l_i is [rows, 8], each tile adds
`sum(reshape(p, [rows, BN/8, 8]), 1)` (in-thread for the MMA layout; TTGIR has no extra local_alloc/barrier, PTX
shfl 23 -> 19 at verify 16 sliding), one reduction in `_finish`.
- Sweeps (2 runs each): every sliding shape flat or +0.1..0.3 us. Paired (tools/s26/abp.py, 9 rounds): decode 1
  sliding +0.19 us (0/9 wins), verify 1 sliding 0.00 (4/9), verify 1 full +0.02 (4/9).
- Why: the Triton shapes are latency-bound gathers; the quad shuffles of the row sum were never on the critical path.
  Don't retry on the Triton path.

New tool: tools/s26/abp.py '<shape>' '<json override>' [rounds]: paired A/B of a plan override on one scored shape.

### Next ideas

Unchanged from sessions 24/25: (1) check main de522ae; (2) verify full's consumer is the only large lever (FP8 P@V
needs a conflict-free V transpose, inline PTX = Codex's lane); (3) session 23 idea 2 (decode 1 full 4-warp `_gd_full`
merging its splits on chip, ~-1 us H100) only if the H200 confirms USE_GD1.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 27 (overnight agent, 2026-10-24 08:53-09:35 UTC)

**Main kernels unchanged (de522ae). Still UNCHECKED on the H200: check main first thing next UTC day, exactly as
session 24's top paragraph says** (verify 16/32 full vs f3e099c1 for gbal, decode 1 full vs 15.21 us for USE_GD1,
decode 16/32 full vs 92.63/173.84 for cw 1). The account still had 0 checks left (10/10 for this problem today).
H100 bench reproduced 2.01x.

### Dead end: sliding verify without key splits (branch exp/rows, plan key `rows`, tools/s27/rows.py there)

The merge launch is 2.3/3.4 us of verify 16/32 sliding (tools/s23/abl.py: 12.48 total / 10.16 no-merge, 19.52 /
16.16). `_attn_rows`: each program takes RM = 64 / rows of one KV head's query rows over the whole window, attends
the new tokens itself (program 0 stores them) and writes the gated output, so there are no partials and no merge.
Correct (10/10 sliding verify cases), but slow: best verify 1/16/32 sliding 20.6 / 21.8 / 32.8 us (rows 4,
block_n 128, 5-8 stages) vs 5.7 / 12.5 / 20.0. Even at verify 1 (8 programs) a program needs ~20 us for 2,048 keys:
with M = 16 rows Triton takes the mma-v2 path (FP8 K and V upcast to FP16, as session 19 found for decode full), at
~8 elements/clk/SM. Pipelining does work (stages 1/2/5 -> 48.8/34.9/21.8 us). Don't retry row splits in Triton;
a native FP8 m16n8k32 kernel (Gluon mma_v2 as in `_gd_full`) would be needed, and K/V are still reread per row group.

### Measured, no change

- Verify 16 sliding is not limited by tiles per program: splits 4/8/16 -> 12.7/12.4/16.1 us (block_n 64). Its
  time is the fixed dependent-load chain (seq -> page table -> rows, q/RoPE, split 0's new tokens) plus the merge.
- Paired (tools/s26/abp.py): verify 16 sliding block_n 32 stages 8 -0.03 us (6/9, noise); verify 32 sliding stages
  8 +1.15 (0/7).

### Next ideas

Unchanged from session 26: (1) check main de522ae; (2) verify full's consumer is still the only large lever (FP8
P@V needs a conflict-free V transpose: inline PTX, Codex's lane); (3) decode 1 full on a 4-warp `_gd_full` merging
its splits on chip, only if the H200 confirms USE_GD1.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 28 (overnight agent, 2026-10-24 09:00-09:40 UTC)

**Main kernels unchanged (de522ae). Still UNCHECKED on the H200: check main first thing next UTC day, exactly as
session 24's top paragraph says** (verify 16/32 full vs f3e099c1 for gbal, decode 1 full vs 15.21 us for USE_GD1,
decode 16/32 full vs 92.63/173.84 for cw 1). The account still had 0 checks left (10/10 for this problem today).
H100 bench reproduced 2.008x.

### Dead end: FP16 V ring decoupled from the FP8 gather ring (branch exp/nv, plan key `nv`)

Session 10 found NS 4 + LAG 2 forces 1 CTA/SM because the FP16 V halves ring had NS slots (32 KB/stage). With `nv`
FP16 slots (< NS) the producer, before converting tile j into slot j % NV, waits on `empty` for tile j - NV (parity
(j - NV) // NS), so NS 4 + NV 2 fits in the same 96 KB as NS 3 today and keeps 2 CTAs/SM.
- Sweep (H100 real): verify 16 full NS4/lag1/nv2 137.5 vs 137.8 base (tie), NS4/lag2 139.2, NS4/lag3 146, NS5 280
  (verify 32; smem pushes it to 1 CTA/SM). Verify 32: base 251.3, NS4/lag1 251.9, NS4/lag2 253.1.
- Paired (tools/s21/ab.py, 5 rounds), all-L2 = the compute-bound proxy for the H200: NS4/lag1/nv2 +3.25 / +6.58 us
  (verify 16/32), NS4/lag2/nv2 +1.30 / +3.87; real +0.03/+0.27 and +1.31/+1.28. The extra wait puts the V conversion
  right behind the consumers' P@V; fewer FP16 slots cost more than the deeper gather ring gains. Deeper rings don't
  help verify full on either regime; don't retry.

### Also looked at (no change)

- Judge time minus profile GPU-busy is ~1.2-2 us for every batch-1/sliding shape across all saved profiles (all
  2-kernel plans), the same as what bench.py's empty-replay subtraction removes, so there's no evidence that the judge
  charges extra for a second kernel node: fusing the merge for its own sake is not a hidden H200 win.

### Next ideas

Unchanged from session 27: (1) check main de522ae; (2) verify full's consumer is the only large lever (FP8 P@V
needs a conflict-free V transpose: inline PTX, Codex's lane); (3) decode 1 full on a 4-warp `_gd_full` merging its
splits on chip, only if the H200 confirms USE_GD1.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 29 (overnight agent, 2026-10-24 09:06-09:50 UTC)

**Main kernels unchanged (de522ae). Still UNCHECKED on the H200: check main first thing next UTC day, exactly as
session 24's top paragraph says** (verify 16/32 full vs f3e099c1 for gbal, decode 1 full vs 15.21 us for USE_GD1,
decode 16/32 full vs 92.63/173.84 for cw 1). The account still had 0 checks left (10/10 for this problem today).

### Dead end: decode 1 full with W warps per CTA, merged on chip (branch exp/gd4, plan key `w4`, tools/s29/)

Session 23's idea 2. `_gd_full4`: the `_gd_full` one-warp chain written with rank-3 tensors (warps on dim 0; Gluon
`mma_v2` takes batched [W, M, K] operands: tools/s29/probe3d.py), each warp its own key split, the CTA merging its
W partials before the store, so the merge reads S + 1 = 128 slots instead of 512. Correct (tools/s29/cases.py: 6
decode-1 full cases, same errors as main). H100 (tools/s23/abl.py, total / no-merge us), base 15.7 / 12.4:
- W 4, 127 CTAs: 16.1-16.6 / 13.6-13.7. Merge itself 3.3 -> 2.4 as predicted, but attention +1.3:
  - the 4-warp layout alone (direct per-warp stores, no exchange): +0.7-0.9;
  - the on-chip exchange (STS acc [W, 16, 128] fp32 + barrier + LDS): +0.5 (`gl.sum` over the warp dim
    emitted 68 `bar.sync`, worse: 14.4); swizzle (xv 4, xp 8) helps a little.
  - not the per-tile barriers: `bar.warp.sync` instead of `gl.barrier()` (wsync) and V loaded straight into the
    dot layout (vd, no convert_layout STS/BAR) both tie.
- Even W 1 through the new code: 13.5 attention (exchange epilogue; 12.6 with direct stores).
- W 2 / 255 CTAs: paired (tools/s26/abp.py, 7 rounds) +0.08 us, 3/7 wins. W 8 / 127: 16.4; W 8 / 63: 17.8;
  W 4 / 255: 16.9; W 4 / 191: 16.2.
The one-warp `_gd_full` + 512-slot merge stays. Don't retry multi-warp CTAs for this kernel.

### Next ideas

Unchanged: (1) check main de522ae (session 24's list); submit if it beats the board (account best 2.1605x); (2)
verify full's consumer is the only large lever left (FP8 P@V needs a conflict-free V transpose: inline PTX, Codex's
lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 30 (overnight agent, 2026-10-24 09:15-09:35 UTC)

**Main 4d7e4e0 = de522ae (session 24's three unchecked changes) + three batch-1 plan changes below, all UNCHECKED on
the H200.** H100 bench 2.013 -> 2.043 (decode 1 sliding 4.86 -> 4.51, verify 1 sliding 5.89 -> 5.60, verify 1 full
18.94 -> 17.95). Each touches its own shape, so the next check evaluates everything at once: besides session 24's list
(verify 16/32 full vs f3e099c1, decode 1 full vs 15.21, decode 16/32 full vs 92.63/173.84), compare **decode 1 sliding
vs 6.35, verify 1 sliding vs 7.42, verify 1 full vs 16.41 us (f3e099c1)**. Fallbacks: verify 1 full worse -> first
revert its block_n 64 / stages 8 (keep 127 splits + tsep); a sliding batch-1 shape worse -> set `tiny` False.

### Change: `tsep` (plan key; Triton `_attn_split`, constexpr SEP)

The block's new tokens (`_new`: norm/RoPE, FP8 store, causal attention) run in an extra program (split index S,
partial slot S, no cached keys) instead of split 0, which was the slowest program. Upper bound measured first by
skipping `_new` (wrong results): batch-1 sliding -0.2..-0.4 us, verify 16 sliding -0.2, verify 32 sliding 0.
- On for batch * hkv <= 4 (sliding, both q_lens) and verify blocks in the `heads <= 4` full branch (verify 1 full).
- Batched sizes lose (verify 16/32 sliding +1.3/+2.7, decode 16 +0.6; with one split fewer so the merge keeps a
  power-of-two slot count: still +0.4..+0.9, decode 32 sliding paired +0.93, 0/11). Off there.
- Merge: `_combine_hb` gets S + 1 slots (sb rounded up to a power of two).

### Plan changes (H100 paired, tools/s26/abp.py)

- Sliding batch 1-2 (`tiny`): tsep + stages 2. Decode 1 sliding: tsep alone tie (-0.03, 6/11), stages 2 alone tie
  (-0.03), **both -0.35..-0.42 us (15/15, 11/11)**. Verify 1 sliding: tsep -0.14 (8/9 twice), stages 2 vs 3 a tie.
  Re-sweep after: warps 2/8, cw 2, hb 1, BF16 partials all worse.
- Verify 1 full: tsep (-0.11, 9/11), 127 splits so the merge has <= 128 slots in one pass (-0.08, 8/11), then
  block_n 64 + 8 stages (-0.62, 9/11). New vs old plan **-0.82 us, 15/15**. Emulated H200 (tools/emu.py 40) 25.9 ->
  20.3 (the emulator overstates this shape; the judge measured 16.41 for the old plan). Warning from session 18: a
  similar batch-1 full change (decode 1 full, hpp 2 / block_n 64) won on the H100 and emulator but lost on the H200.
- Tied / lost: verify 1 full block_n 64 with stages 6/10/12, block_n 32 stages 12, splits 95/191; decode 1 full gdec
  `sep` (+ splits 510) -0.02 (6/11), stages 3 +1.0.
- Dual chains (`dual`) on verify 16/32 sliding: +1.2..+14 us. Dead.

### Next ideas

1. Check main 4d7e4e0 when the UTC day resets (session 24's list + the three shapes above); submit if it beats the
   board (account best 2.1605x).
2. The stage count interacted with tsep (decode 1 sliding): after any structural change to a batch-1 path, re-sweep
   its stages/block_n, which earlier sessions treated as settled.
3. Unchanged: verify full's consumer is the only large lever (FP8 P@V needs a conflict-free V transpose: Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 31 (overnight agent, 2026-10-24 09:31-10:30 UTC)

**Main 8784ba0 = 4d7e4e0 (all of sessions 21-30's changes, UNCHECKED on the H200) + two merge-kernel changes below,
also unchecked.** H100 bench 2.031 (start) -> 2.057. 0 judge runs (account had 0 checks left today). Next UTC day:
check main first; compare per shape with session 24's and session 30's lists (f3e099c1: decode 1 sliding 6.35,
decode 1 full 15.21 (beb63898), verify 1 sliding 7.42, verify 1 full 16.41, verify 16/32 full 124.65/231.75, decode
16/32 full 92.63/173.84). Both changes below can be flipped off at once with `MERGE1 = False`.

### Change 1 (c677903): one-pass merges load everything before the max reduction (`MERGE1`, constexpr `ONE`)

`_combine` / `_combine_hb` with all slots in one pass (S_PAD <= SB) used to load m/l, reduce the max, then load m
again and the accumulators: two dependent L2 round trips. Now m, l and the [S, D] accumulators are loaded together.
H100 paired (tools/s31/ab.py, 11 rounds): decode 1 full -0.22 (11/11), verify 1 full -0.16 (10/11), decode 32 sliding
-0.10 (9/11; recheck -0.05 6/11), verify 16/32 sliding -0.10/-0.08, decode 16 sliding / decode 16 full / verify 16 full
~0. Decode 1 sliding +0.05 (2/11): the tsep merge pads 33 slots to 64, so it keeps the old path (`ONE=... and not sep`).

### Change 2 (8784ba0): `_combine_bal` (gbal merge, verify 16/32 full) as one online pass

It made a max pass over its programs' partials and then a second pass loading them again. Now each chunk's m, l and
accumulators load together with an online rescale (running max, alpha). **H100 paired verify 16/32 full -1.23 / -0.98 us,
11/11 each.** Tests 16/16, gluon_cases 8/8, nearby 48/48.

Lesson: the merge kernels are pure latency chains; look for dependent loads that need not be dependent. Other merge
variants for verify 1 full (hb 2/4 with cw 1-8) were all worse than the default `_combine`.

### Next ideas

1. Check main 8784ba0 next UTC day; submit if it beats the board (account best 2.1605x).
2. The same "issue loads early" lens on the attention kernels' prologues (seq -> page table -> rows chain, q/RoPE)
   and on the Gluon in-grid merge (gfuse) for verify 8-12 full.
3. Unchanged: verify full's consumer (FP8 P@V needs a conflict-free V transpose: Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 32 (overnight agent, 2026-10-24 09:44-10:10 UTC)

**Main = 8784ba0 + `pf` for verify 1 full (below). Everything since session 21 is still UNCHECKED on the H200** (account
had 0 checks left today). Next UTC day: check main first and compare per shape with session 31's list; verify 1 full
was 16.41 us on f3e099c1. H100 bench 2.05 (verify 1 full 17.89 -> 17.58; other shapes unchanged within noise).

### Change: `pf` (plan key, Triton `_attn_split` constexpr PF; on for verify 1 full only)

Before the seq/query loads, each program loads one page-table entry per 128-byte line of its own split's chunk
(split * CHUNK onward; full layers' fixed chunks, so no seq dependence) and folds their sum into m_i at the end
(`+ where(sum == -7, 1, 0) * 0.0`, so nothing stalls on it). The pipeline prologue's page loads then hit L2.
**H100 paired -0.35 us (10/11), -0.37 (8/11), off-vs-on recheck -0.30 (10/11).** Tests 16/16, nearby 48/48.

How it was found (TTGIR, tools/s32/): the warps stall on the query load (RMSNorm before the loop; LICM hoists any
in-loop norm back above the prologue) and only then issue the prologue's page-table async copies. Upper bounds on
verify 1 full: queries replaced by a constant -0.35..-0.6 us; seq-independent loop bounds (`fixed`, masked loads,
clamped page index) 0.0 with or without the constant queries, so the seq hop is not on the critical path, the query
hop is.

### Dead ends (this session)

- `fixed` (seq-independent loop over the whole chunk) and `lq` (queries normalized inside the loop): tie / -0.08.
  LICM hoists the in-loop norm.
- `pf` + also touching the first tile's K/V rows (pf 2): tie. `pf` on sliding layers (chunk from seq - window):
  decode 1 sliding +0.26, verify 1 sliding -0.08, decode 16 sliding +0.03: off.
- Same page-line prefetch in Gluon `_gd_full` (decode 1 full; range estimated from the table width): tie (-0.02,
  5/11). Reverted.
- RoPE cos/sin computed in-kernel instead of the table load: tl.cos/tl.sin (libdevice) make decode 1 sliding 4.5 ->
  5.8 and verify 1 sliding 5.6 -> 18.7 us; upper bound with no cos/sin load at all: 0 gain. Not on the critical path.
- Verify 1 full re-sweep with pf: stages 6/7/9/10, block_n 32/12 stages, block_n 128/4, splits 95/159, warps 8: all
  tie or worse (stages 8 is a sharp optimum: 6 and 10 both +0.7..1.0 us).

### Floors (H100, tools/s32/gather_sliding.py, event-timed incl. launch)

A synthetic gather of 32 x 2048 random 128-byte K+V rows (decode 32 sliding's data) takes 15.3 us best; our decode 32
sliding is 15.7 (bench). With bench's profile (tools/s32/prof.py), verify 1 full attention 16.3 and decode 1 full 14.4
us are near the same per-row rate. On the H100 every scored shape is now within ~10-20% of a random-row gather floor;
the remaining lever is the H200's compute-bound verify full consumer (Codex's lane: FP8 P@V).

### Next ideas

1. Check main next UTC day (session 31's list + verify 1 full); submit if it beats the board (account best 2.1605x).
2. If the H200 shows a shape far from H100-proportional, look there; on the H100 the latency hunting is near done.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 33 (overnight agent, 2026-10-24 09:57-10:20 UTC)

**Main: my code unchanged; the human's b94e35f (pushed during this session) put split partials back to FP32 everywhere and CLAUDE.md now forbids BF16 partials (below SGLang's precision): their bench 2.059 -> 2.036x. The H100 numbers below were measured before it (BF16).** Everything since session 21 is still UNCHECKED on the H200.

**For the human:** the board's scored runs 10d12495 (2.1605x, account best) and beb63898 (2.138x) used BF16 partials (session 15 onward). I did not withdraw them (your change didn't); if they should come off the board, `okc withdraw <job>`. Compare future checks against an FP32-partials baseline, not 2.1605x. 0 judge runs
(account at 10/10 checks today). H100 bench at start 2.071 (decode 1 sliding 4.48, verify 1 sliding 5.26, verify 1
full 17.23, verify 16/32 full 136.2/250.5). Next UTC day: check main first (session 31's per-shape list + verify 1 full
16.41 on f3e099c1); submit if it beats the board (account best 2.1605x; board leader now 2.726x).

Where the gap is (f3e099c1 per-load speedups): decode 1 2.52, decode 16 2.11 (2.27 with beb63898's sliding), decode
32 2.17, verify 1 2.70, **verify 16 1.81, verify 32 1.72**. The verify loads' full layers (H200 124.65/231.75 us,
compute-bound there) are the lever; getting them to decode-full speed (~93/175) would be about +7% overall.

### Dead end 1 (branch exp/pfr, not pushed): page-table prefetch on batch-1 sliding (plan `pfr`)

Upper bound first: skipping the page-table load (page = token index, wrong results) makes decode/verify 1 sliding
~0.45 us faster, so the seq -> page-table hop is on the critical path. But the window's place in the row depends on
seq, so a seq-independent prefetch has to cover the whole row (80001 entries, 320 KB):
- per-line `tl.load`s: decode 1 +0.78, verify 1 +0.53; per-line `prefetch.global.L2`: +0.21 / 0.0;
- one `cp.async.bulk.prefetch.L2` per program (row split over the 66 programs): decode 1 +0.26 (0/15), verify 1
  -0.06..-0.16 at first, then off-vs-on recheck +0.03 (6/15): noise. Batch 16 sliding +2 us. A 16-byte bulk
  prefetch ties, so the loss is the extra DRAM traffic.

### Dead end 2 (branch exp/qreg, pushed): gbal verify-full queries as QK^T's register A operand (plan `qreg`)

Idea: skip the per-tile shared-memory read of the 64x64 FP8 queries (A operand from registers, 8 regs/thread).
- Held across the tile loop it gives NaN: in SASS, ptxas puts the query fragment in R88-R95 and the loop's F2FP (P to
  FP16) overwrites them after the first tile (the PTX keeps the virtual register live; a non-pure `mov` copy doesn't
  help). Unit tests (tools/s33/qreg_unit*.py, qreg_ws.py: register-A FP8 wgmma, also in warp_specialize partitions
  and on the 128B-swizzled K slice) are all correct; it's the in-loop register allocation. The broken kernel timed
  all-L2 -2.4/-5.1 us at cregs 104 / pregs 32 (+9/+16 at 96), but that doesn't count.
- Correct variants: per-tile `q_s.load(DotOperandLayout)` all-L2 -0.4/-0.75 (tie), real tie; held queries
  with each tile's operand `qa ^ loop-variant zero` (inline asm) +2.5/+4.9. Dead.
- Gluon facts: `warpgroup_mma_wait(deps=[fp8 tensor])` fails to compile ("Unexpected bitwidth in WGMMAWaitGroupOp:
  8"); FP8 register A operands work with DotOperandLayout(0, parent=wgmma layout, k_width=4).
- Debug tools: tools/s33/dbg.py (non-finite count per batch/head/token for a plan override on a gbal case),
  tools/s33/dump.py (on exp/qreg: a kernel debug buffer comparing two score paths on one tile).

### Next ideas

1. Check main next UTC day; submit if it beats the board.
2. Verify 16/32 full on the H200 is the lever (above). On the H100, compute-only (all-L2) is the proxy;
   the consumer has no register headroom at 2 CTAs/SM, so ideas that add live state lose. Unchanged: FP8 P@V with a
   conflict-free V transpose (Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 6/10 scored.

## Session 34 (overnight agent, 2026-10-24 10:26-10:50 UTC)

**Main unchanged (24d7ae8, code = b94e35f: FP32 partials everywhere).** H100 tests 16/16, bench 2.016x at start (the
human measured 2.036x on b94e35f; run-to-run noise on this sandbox is ~1%). 0 judge runs: the account has 0 checks
left today (`okc problems`: checks 10/10 used, scored 7/10; the human's submit abd67ae2 on b94e35f was pending).
Goal was to win back FP32 partials' cost (decode 1 full +0.5, verify 1 full +0.8, verify 32 sliding +0.7 us). Nothing
found: on the H100 the FP32 cost is intrinsic to the bytes the merges read. New tools: tools/s34/abp.py (paired A/B of
any number of plan overrides on one shape), tools/s34/cmp.py (output of plan B vs A over repeated calls and 200 graph
replays).

### Dead ends (H100, paired, medians)

- Plan re-sweeps with FP32 partials, all tie or lose vs the current plan: decode 1 full splits 127/191/255/383 (+1.0
  to +4 us), csb 256 (+1.0), cw 4 (+1.15), cw 16 (+0.1); verify 1 full splits 63/95 (+1.8/+4), cw 2/8, hb 2, csb 64
  (0..+0.3); verify 32 sliding splits 4/6/16, fuse (+3.8), hb 1/4/8, stages 4/6; decode 16/32 full splits 47/79 and
  23/39, cw 2. The merge configs from the BF16 days are still optimal.
- **D-split merge for decode 1 full** (grid (1, 16 heads, DS); each program merges D/DS columns of all 512 slots, so
  64 SMs pull the 2 MB of FP32 partials instead of 16). Upper bound with a wrong (per-chunk) RMSNorm: -0.6 us (ds 4, cw
  4, 7/7). Made correct by exchanging the chunks' sums of squares (store, acq_rel arrival on a count-up counter, spin,
  load): bit-exact, but +0.66 us (0/11). The exchange is about 1.2 us, i.e. any cross-program reduction costs about an
  L2 round trip, which is as large as what it saves. (Gotcha: count-up counters shared between different DS values in
  one process hang; key them by everything that changes the arrivals per call.)
- **Flag-driven merge instead of the PDL wait** (tools/s34/flagmerge.diff): each _gd_full program bumps its own
  ready word (release), and the merge polls the words and skips griddepcontrol.wait (the wait was moved to its end).
  Upper bound, merge with no wait at all (racy): -0.66 us (7/7). Real: merging chunks of 32-256 slots online as they
  became ready is +2.4..+15 us (two serial L2 round trips per chunk); polling all 512 words then one-pass loading is
  +0.9 (cw 8), +1.3 (cw 16), and with nanosleep backoff +1.0/+1.1 (so the cost isn't contention from spinning). PDL's
  grid-completion wait is faster than release + poll.

### Next ideas

1. Check main (b94e35f code) the next UTC day if the human's abd67ae2 hasn't already measured it; compare per shape
   with session 31's list. FP32 partials are the baseline now: compare against abd67ae2, not 2.1605x.
2. The merges are at their latency floor on the H100; don't spend more time on merge plumbing. Verify 16/32 full on
   the H200 (compute-bound consumer) is still the main lever (FP8 P@V with a conflict-free V transpose; FA3 itself
   feeds P to the PV wgmma in FP8 for FP8 caches, so that's at SGLang's precision. Inline PTX parts are Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 35 (overnight agent, 2026-10-24 10:46-11:25 UTC)

**Main code unchanged (b94e35f, FP32 partials).** 0 judge runs: the account has 0 checks left today (10/10 used,
scored 7/10). Board best is ours: the human's abd67ae2 = 2.263x [2.229, 2.296] on b94e35f.

H200 per shape from abd67ae2 (ours, us, speedup): decode 1 sliding 6.17 (3.45x), full 14.16 (2.20x); decode 16 10.27
(2.63x) / 90.3 (2.21x); decode 32 14.25 (2.19x) / 172.8 (2.22x); verify 1 7.20 (3.18x) / 17.18 (2.08x); verify 16
12.62 (2.31x) / 117.1 (1.74x); verify 32 19.05 (1.81x) / 216.4 (1.79x). Weighted load times (46 x sliding + 16 x
full): verify 32 = 876 + 3462 us, so its full layer is 80% of that load.

### Dead ends: the gbal verify-full consumer (branch exp/pexp, pushed; plan key `exa`, tools/s35/sass.py dumps SASS)

Paired H100 medians, verify 32 full real / all-L2 (tools/s21/ab2.py), base 250.9 / 196.0:
- `exa 1` (no exp2 for P, wrong results): -0.6 / -5.9 (verify 16: -0.2 / -2.9). MUFU isn't the limit, so a
  polynomial (FMA-pipe) exp2 for part of P can win at most ~1.5% of the all-L2 time. Not built.
- `exa 3` (producer issues no gathers and no V conversion, only the barrier handshakes): 173.8 / 172.9. The
  consumer alone is ~2050 cycles per 64-key tile per warpgroup (151 tiles per program at batch 32).
- `exa 4` (no V conversion only): 249.7 / 182.1.
- `exa 5` (row sums on the tensor core as P @ ones[64, 8], FP32 accumulate; drops the 30 FADD per tile): correct
  (tools/s21/cases.py 8/8), but -0.35 / +0.24, verify 16 +0.18 / +0.14: a tie, the same result as session 14's table.
- Consumer fast path per thread per tile (SASS): 36 FMNMX, 36 FFMA, 34 MUFU.EX2, 34 FMUL, 30 FADD, 16 F2FP, 2 QGMMA,
  4 HGMMA, about 240 instructions. Taking out FADDs or MUFUs does nothing: it's the serial QK -> softmax -> P@V chain
  with only 4 consumer warpgroups per SM (register-limited at 2 CTAs/SM).
- (`exa 2`, consumer skipping its tile work with `continue`, doesn't compile in Gluon; not needed.)

Before trying anything on this consumer, read session 14's dead-end table (FA3 QK(i+1) pipelining, FP16 scores,
P @ ones, conditional rescale) and session 22 (FP8 P@V, at most ~0.3%). None of it is left untried.

### Next ideas

1. Next UTC day (from 00:00 UTC): main == the checked b94e35f, so there's nothing new to check unless a change wins.
2. Untried and cheap, low odds: BN 32 tiles with QK(i+1) issued during softmax(i) (two 16-register S tiles fit in
   the budget one BN-64 tile uses now). Session 14 timed BN 32 unpipelined at +14 us, so the per-tile overhead would
   have to be won back first.
3. Batch-1 sliding has the most weight per us: 1 us on decode 1 sliding is about 9% of that load, ~1.5% overall. The
   H200 times are 1.7 us above the H100 bench (6.17 vs 4.48); see whether that gap is launch/graph cost we can cut.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 36 (overnight agent, 2026-10-24 10:55-11:35 UTC)

**Main code unchanged (b94e35f, FP32 partials; checked as abd67ae2 2.263x).** 0 judge runs (account checks 10/10 used
today; the per-day check limit on this problem is 10 now, not 20). H100 bench at start 2.034x (decode 1 sliding 4.58,
verify 1 sliding 5.62, decode 1 full 15.76, verify 1 full 18.18, verify 16/32 full 135.4/250.5). Added the human's
abd67ae2 profile files to the repo (profiles/abd67ae22a0743b0.*). New tools: tools/s36/abc.py (paired A/B of two
submission *files* on scored shapes, for code changes that aren't plan keys), tools/s36/empty.py (graph fixed costs).

### Why batch-1 sliding is ~1.6 us slower on the H200 than on the H100 bench (session 35's idea 3): accounting

- H100 vs H200 (abd67ae2): decode 1 / verify 1 sliding 4.58/5.62 vs 6.17/7.20 (+1.6), but decode 16/32 and verify
  16/32 sliding are 0.2-1.6 us *faster* on the H200, as are decode 1 full and verify 1 full (-1.0..-1.6).
- H100 graph replays: empty graph (no kernel) 2.88 us, one tiny kernel 5.18, two tiny kernels 6.51. bench.py subtracts
  the one-kernel graph (5.2 us); the judge subtracts "about 4 us". So the judge's times probably keep ~1-1.5 us of
  the first kernel's launch that bench.py takes off. Batch-1 sliding is pure latency and doesn't gain from the
  H200's bandwidth, so the offset shows there; the bandwidth-heavier shapes gain more than it costs.
- The judge's profile (eager, no PDL overlap at these sizes): decode 1 sliding _attn_split 2.9 + _combine_hb 1.8 us;
  H100 eager without PDL 4.16 + 1.94 (bench.py's flush *writes* 512 MB, so the H100 eager attention also pays dirty
  L2 write-backs; the judge only reads its flush buffer).
- The only lever is fewer kernels, and that's dead: fused last-program merge on batch-1 sliding is 10.0-11.2 us vs
  4.46 (decode 1) and 14.7-17.3 vs 5.66 (verify 1); in-grid/cluster merges were already measured at the PDL merge's
  cost (sessions 21, 25, 34). Don't chase this gap again.

### Dead ends (H100)

- Decode 1 sliding plan re-sweep with FP32 partials (splits 16/32/64 x warps 2/4/8 x block_n 32/64 x hpp 1/2, 31
  plans): the current plan (splits 32, block_n 64, 4 warps, stages 2, tsep) is best; next 4.77-5.65 vs 4.58.
- Merges loading the V scale before the PDL wait (like the gate already is), in _combine/_combine_hb/_combine_bal:
  paired 11 rounds, decode 1 sliding -0.03, verify 1 sliding 0.00, decode 1 full +0.02, verify 1 full -0.13 (7/11),
  decode 16 sliding -0.05, verify 32 sliding +0.10: a tie (the attention kernel reads vs too, so it's an L2 hit).
  Not kept.

### Next ideas

1. Next UTC day: main == checked b94e35f, nothing new to check. Only check a change that wins clearly on the H100.
2. The H100 latency hunting is done (sessions 25-36). Verify 16/32 full on the H200 (compute-bound Gluon consumer) is
   the remaining lever and every Triton/Gluon-level variant of it has been tried (sessions 14, 22, 33, 35 tables).
   Untried, low odds: BN-32 tiles with QK(i+1) issued during softmax(i). Anything with inline PTX / native code is
   Codex's lane.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 37 (overnight agent, 2026-10-24 11:02-11:50 UTC)

**Main code unchanged (b94e35f, FP32 partials; checked as abd67ae2 2.263x, rank 11; board #1 is now 2.726x, #10
2.343x).** 0 judge runs (account 10/10 checks used today, scored 7/10; Claude's 3 scored for today are used).
Tried session 35/36's last untried consumer idea: BN-32 granularity with QK^T overlapping the softmax, in the gbal
verify-full consumer. Dead. Branch exp/bn32 (pushed; plan key `pipe`, helper `_gw_halves`); tools/s37/sass.py
(SASS + STL/LDL/GMMA/DEPBAR counts) and tools/s37/ptx.py (dump PTX; run `ptxas -arch=sm_90a -v` on it to get
ptxas's wgmma serialization warnings; cuobjdump/ptxas are in /usr/local/cuda-13.0/bin, not on PATH).

### Dead ends (H100, tools/s21/ab2.py, 5 rounds, verify 32 / 16 full, real and all-L2; base 251 / 136.5 real,
195.5 / 102 all-L2)

- `pipe 1`, FA3-style across iterations (QK(j+1) of the next 32-key half issued before softmax(j), P@V(j-1) waited
  with wait(1)): ptxas **C7514 "wgmma.mma_async instructions are serialized due to non wgmma instructions reading
  accumulator registers ... between start and end of the pipeline stage"** puts `DEPBAR.LE gsb0, 0` after every
  wgmma (even splitting one QK into two waits): +24 / +74 us. Cause: wgmmas still in flight across the loop
  back-edge (loop-carried accumulator tokens). Anyone pipelining wgmma in Gluon: keep nothing in flight across the
  back-edge, and check with ptxas -v.
- `pipe 1`, rewritten in-tile (two N=32 QK^T per 64-key tile; QK_b during softmax_a, P@V_a during softmax_b, per-half
  max/rescale): no serialization, 8/8 cases, but real +1.7 / +1.7, all-L2 +25.0 / +12.1.
- `pipe 2` (two N=32 QK^T, one max/rescale per tile, exp of half b while P@V of half a runs): real -0.10 / -0.02
  (tie), all-L2 +4.75 / +4.32. Splitting QK^T into N=32 wgmmas costs more than the overlap gives; session 35 already
  showed all of the exp2 is worth only ~3% all-L2, so overlapping half of it can't pay.

### Next ideas

1. The Triton/Gluon verify-full consumer is exhausted (sessions 12, 14, 22, 33, 35, 37). Remaining levers need
   native code (Codex's lane) or a new structural idea, not another consumer variant.
2. Next UTC day: main == the checked b94e35f, nothing new to check.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 38 (overnight agent, 2026-10-24 11:10-~11:35 UTC)

**Main code unchanged (b94e35f, FP32 partials; checked as abd67ae2 2.263x, rank 11; board #1 2.726x).** 0 judge runs
(account checks 10/10 used today, scored 7/10). Focus: why verify 1 full is 12.7 us of attention on the H200 when
decode 1 full (same bytes) is 8.0. Two dead ends (branches pushed), plus a breakdown. New tools in tools/s38: attn_only.py
(whole vs attention-only, merges as no-ops, plan overrides), swf.py / gattn.py (same for any submission *file*, e.g.
ablation copies), mkabl.py (ablation copies in /tmp/s38), chk.py (reference check of a plan override on scored shapes),
sass.py (SASS/PTX + op counts of _attn_split), one.py (runs a shape 3x for ncu). ncu works here:
`/usr/local/cuda-13.0/bin/ncu --kernel-name regex:"_gw|_attn" --launch-skip 2 --launch-count 1 ... python tools/s38/one.py
submission.py "verify 1 full"` (it locks clocks at 1.44 GHz unless `--clock-control none`; `--page source --print-source
sass --csv` gives per-instruction stall samples).

### Verify 1 full breakdown (H100, Triton _attn_split, 127 splits, attention only = 15.4 us; whole 18.2)

- No tiles (hi = lo): 2.0 us (2.2 without pf). Grows with the program count: 510 programs 3.4, 1022 programs 6.2. That
  is the FP32 partial stores (16 KB per program) and prologues. This is why more splits don't pay.
- All K/V reads from L2 (rows page % 4096, so they're spread over slices): 12.7 us. Real minus all-L2 is only 2.7 us,
  so **verify 1 full is compute-chain-bound, not gather-bound**. (Warning: page % 64 instead hot-spots a few L2
  slices and reads 15.5 us. Don't use that as a no-gather ablation.)
- Half the keys per program: all-L2 12.7 -> 8.0, real 15.4 -> 10.2. So ~1.0 us per 64-key tile per program
  (~1900 cycles), at 2 programs (warpgroups) per SM, and an intercept of ~5 us real / ~3.3 all-L2.
- Gluon _gw_verify at batch 1 (non-gbal, LAG 1, splits 127, one CTA = 2 consumer WGs per SM): attention 15.3-16.3, the
  same per-tile slope (1.07 us/tile). At 255 splits (4 consumer WGs/SM), 1.73 us/tile and a larger intercept. gbal at
  batch 1: attention 16.3, and its merge is very slow there (49-153 us whole).
- ncu stall samples (Triton, real): barrier at the loop-top cp.async wait (long_sb), wgmma waits, FENCE/ARRIVE
  (long_sb), MUFU/short_sb; issue slots 31% busy, 7.5 warps/SM. The masked loop runs every tile (verify 1 full has no
  peel), but `peel` ties (15.49 vs 15.52).

### Dead ends (H100)

- FP8 P@V in Triton `_tile` (p.to(e4m3) @ v as stored): verify 1 full 28.3 vs 18.2 us (Triton must transpose V for
  the K-major FP8 B operand). Not kept.
- **Quad consumers** (branch exp/quad, plan key `quad`): `_gw_verify` with 4 consumer warpgroups per CTA (2 per KV
  head on alternate tiles, NS even so each ring slot keeps one parity; each consumer stores its own partial). Correct.
  Verify 1 full attention, 127 splits: 18.4 (LAG 1, pw 4), 17.5 (pw 8), 18.3 (consumers convert their own V, non-LAG)
  vs 16.0-16.3 for the 2-consumer Gluon kernel and 15.4 for Triton main. More consumers per CTA don't speed up the
  per-tile chain. Dead.
- **Spin merge for decode full** (branch exp/spin, plan key `spin`): each `_gd_full` program bumps a per-(request, KV
  pair) counter (release) and `_combine` spins on it instead of griddepcontrol.wait, so short agentic requests merge
  early. Last merge program resets. Correct, but paired 9 rounds: decode 16 full +0.69, decode 32 full +1.36, decode 1
  full +1.71 us (0/9 each). Same finding as session 34's flag merge: the PDL grid wait beats release + poll.
- H100 merge tails (whole - attention only): decode 16/32 full 3.3/2.7 us, verify 16/32 full 3.6/4.0.

### Next ideas

1. Verify 1 full: the per-tile chain (~1 us per 64-key tile per warpgroup, both Triton and Gluon) and the ~5 us intercept
   are the costs. Neither more programs (partials) nor more consumers per CTA helps. What's left needs a shorter
   chain (native code, Codex's lane).
2. Next UTC day: main == the checked b94e35f, nothing new to check.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 39 (overnight agent, 2026-10-24 11:30-~12:05 UTC; unattended session 6 of this night)

**Main code unchanged (b94e35f, FP32 partials; checked as abd67ae2 2.263x, rank 11; board #1 2.726x, #10 2.343x).**
0 judge runs (account checks 10/10 used today, scored 7/10). One new idea tested on the gather microbenchmark,
dead; tools in tools/s39 (gsort.py, gdeep.py, both take the GPU lock).

### Dead end: reading each program's keys in sorted physical order (full layers)

Idea: softmax is order-invariant, and reference.py's pool is a randperm of batch x max_pages split into requests,
so if every program sorted its chunk's rows, programs would sweep memory together and DRAM locality might rise.
H100, 32 x 80000 random 128-byte K + V rows (tools/s39/gsort.py `<chunks> <modes>`):
- chunk 640 / 2560 / 5120: random 239.3 / 236.7 / 235.6 us, sorted 237.6 / 228.8 / 226.1 (-1..-4%, before paying
  for any sort). A lockstep sweep (program p reads rows p, p+P, ...) 229.9; plain arange 241.3.
- **The gather ceiling is the DRAM ceiling**: a dense torch read of the same 655 MB (int64 sum or amax,
  tools/s39/gdeep.py) is 232.5 us = 2.82 TB/s on this H100; random rows reach 236-238 (deeper in-flight configs,
  block 16-128 x 1-8 warps x 4-8 stages, change nothing). Access order and in-flight depth have no headroom; any
  full-layer gain on the H100 has to come from compute or merge tails, not the gather.
- (H200: decode 32 full is 172.8 us for ~606 MB, ~3.5 TB/s. If its dense read is ~4.0-4.3 TB/s the bandwidth bound
  is ~140-150 us, so on the H200 decode full still has ~15% from compute/overlap, as session 19 found.)

### Next ideas

1. Unchanged: the Triton/Gluon consumers and gathers are exhausted (sessions 12-39). What's left needs native code
   (Codex's lane: FP8 P@V with an ldmatrix.trans V transpose, shorter per-tile chains) or a new structural idea.
2. Next UTC day: main == the checked b94e35f, nothing new to check.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 40 (overnight agent, 2026-10-24 ~11:40-12:30 UTC; unattended session 7 of this night)

**Main 533416d = b94e35f + `q16` in the Gluon decode-full kernel `_gd_full` (UNCHECKED on the H200).** 0 judge runs
(account checks 10/10 used today, scored 7/10; Claude's 3 scored are used). Board best is still ours, abd67ae2 2.263x.

### Finding: on sm_90, FP8 mma.sync is emulated by ptxas

ncu on `_gd_full`, decode 32 full with all K/V reads from L2 (new tools/s40/one_l2.py + tools/s40/ops.py, which
aggregates an `ncu --section SourceCounters --page source --print-source sass --csv` dump by opcode): F2FP was 23% of
instructions and FADD 12%. Gluon's `mma_v2` with E4M3 operands is lowered by ptxas to FP16 HMMAs plus FADDs, converting
**both** operands every tile, including the loop-invariant queries. 8 warps/SM (242 registers), issue slots 46% busy,
no pipe saturated (ALU 38%, tensor 24%): latency-bound at low occupancy, so fewer instructions help a little.

### Win (main 533416d): `q16`, default on

Queries converted to FP16 once before the loop (they're E4M3 values, so it's exact), and K loaded as FP8 in the same
dot-operand layout (k_width 4) and upcast with `.to(float16)`. FP16 mma_v2 accepts k_width 4. The results are
bit-identical (same errors in tools/s19/cases.py), with 13% fewer instructions (FADD 12% -> 3%). H100 paired
(tools/s19/ab2.py, 3-4 rounds):
- all-L2 (the H200's compute-bound regime): decode 16/32 full 69.2/133.8 -> 67.7/129.1 us, B64 266.1 -> 255.8, B8
  39.1 -> 38.1, B1 10.6 -> 10.2.
- real: decode 16/32 full 128.7/241.8 -> 128.4/241.3 (tie-ish); emulated H200 (92 SMs) 131.3/251.4 -> 130.6/249.4.
- Tests 16/16, cases 9/9, bench 2.036-2.045 (noise on untouched shapes). Expected on the H200: decode 16/32 full maybe
  -1..-4 us (~0.2-0.4% overall). Check it the next UTC day against abd67ae2: decode 16/32 full 90.3/172.8, decode 1 full
  14.16. If any is worse, `q16: False` in the plan restores the old code path.

### Dead ends

- **Transposed QK^T** in `_gd_full` (branch exp/tqk, plan key `tqk`, pushed): per head S^T = K [BN, D] @ Q^T [D, G]
  (A = K from smem, B = Q^T via smem; half the MMAs of the block-diagonal form), softmax over dim 0, then TPV's
  transposed P@V. Correct (9/9), but all-L2 decode 16/32 79.4/149.6 vs 68.6/131.3 (tpv alone 75.7/142.9). The MMAs
  aren't the cost; the column reductions and P^T layout conversions are.
- **Same trick in Triton `_attn_split`** (queries E4M3-rounded then FP16, K upcast, `QK8 and not QF` to `_tile`/`_new`):
  paired 9 rounds (tools/s36/abc.py), worse on every Triton shape: decode 1/16/32 sliding +0.22/+0.13/+0.16, verify
  1/16/32 sliding +0.18/+0.75/+1.09, verify 1 full +0.98 us. Triton's own FP8 dot lowering is better than an explicit
  upcast there. Not kept.
- **One 128-byte row per thread for the `_gd_full` gathers** (BlockedLayout [1,16] x [32,1], so one row address per
  thread per tile instead of 8): 1.5-2.4x slower (decode 32 full 570 vs 242 us). The cp.async pattern matters much more
  than the address arithmetic. Not kept.

### Next ideas

1. Next UTC day: check main 533416d (q16) on the H200; compare decode 1/16/32 full with abd67ae2 (14.16/90.3/172.8).
2. `_gd_full` remaining instruction mix (q16, all-L2): F2FP 22% (K and V upcasts, 64 each per 32-key tile, plus P),
   FMUL 11% (acc * alpha over the block-diagonal [16, 128] accumulator, half of it unused columns), HMMA 10%, IMAD 8%.
   A cheaper V path would need FP8 P@V with a transposed V (no b8 ldmatrix.trans on sm_90), which is Codex's lane.
   Register use (242, so 8 warps/SM) caps latency hiding: a lighter accumulator (TPV-style [D, G] per head without TPV's
   staging costs) is the only Triton/Gluon-side lever left that I can see.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 41 (overnight agent, 2026-10-24 11:49-~12:25 UTC; unattended session 8 of this night)

**Main 63a4974 = 533416d (q16) + `kst` in the Gluon decode-full kernel `_gd_full` (both UNCHECKED on the H200).**
0 judge runs (account checks 10/10 used today, scored 7/10; Claude's 3 scored used). Board best still ours, abd67ae2 2.263x.

### Win (main 63a4974): `kst`, heads stacked along the P@V reduction (default on, plan key `kst`)

The block-diagonal form computed acc [16, 128] = P [16, BN] @ V [BN, 128] and kept only each row's own head's 64 columns
(half the accumulator and the `acc * alpha` FMULs were wasted). Now acc [16, 64] = P_0 @ V_0 + P_1 @ V_1, where P_j is P
with the other head's rows zeroed (`gl.where` on the PA operand) and V_0/V_1 the two 64-column halves. Same HMMA count,
half the accumulator registers and alpha FMULs, half the partial store.
- How V is split matters: **load the FP16 V tile into VB as before, then `reshape [BN, 2, D] -> permute -> split` and
  `convert_layout` each half to VB** (no-op-ish). Loading the two halves from smem separately (`v_ring.slice(0, D, 1)`)
  makes ptxas shuffle B-operand registers (IMAD.MOV/MOV +7% of instructions): all-L2 +3%, worse than baseline.
- Bit-identical (tools/s19/cases.py: same errors to 4 digits; tools/s41/kcases.py runs cases.py with plan overrides).
  Tests 16/16. Bench 2.042 (noise on other shapes).
- H100 paired (tools/s19/ab2.py, 3 rounds): all-L2 decode 1/8/16/32/64 full 10.1/37.6/66.7/126.7/253.9 ->
  9.7/36.3/63.4/120.4/241.8 us (-4..-5%); real 15.5/74.0/128.5/241.8/467.0 -> 15.5/74.1/128.3/241.6/465.8 (tie, B64 -1);
  emulated H200 (92 SMs) tie. On the H200, where decode full is ~15% compute-bound (sessions 19/39), expect maybe
  -2..-6 us on decode 16/32 full, together with q16.

### Next ideas

1. **Next UTC day: check main 63a4974 on the H200** against abd67ae2: decode 1/16/32 full 14.16/90.3/172.8 us. If
   worse, `kst: False` then `q16: False` in the decode-full plans restore the old paths. If it beats the board, submit.
2. Gluon sliding decode (`USE_GSLIDE`, off since check f3e099c1 lost on the H200) now also gets q16 + kst; it lost by
   3 us at decode 16 sliding, which looks like more than compute, so probably still a loss, but it is a cheap H100 A/B.
3. No Triton scored path uses hpp=2 block-diagonal accumulators now, so `kst` has no Triton target.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 42 (overnight agent, 2026-10-24 11:55-~12:55 UTC; unattended session 9 of this night)

**Main code unchanged (63a4974 = q16 + kst in `_gd_full`, UNCHECKED on the H200; last checked code b94e35f =
abd67ae2 2.263x).** 0 judge runs (account checks 10/10 used today; Claude's 3 scored used). Board: #1 2.726x, we're #11.

### Gluon sliding decode (USE_GSLIDE) with q16 + kst: still an H100 win, still unverifiable

Paired 7 rounds (tools/s36/abc.py, main vs a copy with `USE_GSLIDE = True`): decode 16/32 sliding -1.28/-2.46 us,
7/7 each. The old H200 loss (check f3e099c1: 10.36 -> 13.56 us at decode 16) is NOT in the kernels: that check's
profile shows _gd_full + _combine_hb 6.6 + 2.3 = 8.9 us vs Triton 7.1 + 1.7 = 8.8 in abd67ae2, yet the scored time was
+3.2 us. So ~3 us goes somewhere outside the profiled kernel time (cold-L2 effects? graph structure?). Worth one check
the next UTC day only if a check is spare: main with USE_GSLIDE = True (main already has q16 + kst to check anyway, so
check main first and USE_GSLIDE separately or not at all).

### Verify-full consumer occupancy (the H200's compute-bound regime): measured scaling, then a dead end

- **Consumer throughput scales ~linearly with consumer warpgroups per SM**: plan key `cps` (programs per SM in
  `_verify_bal`, branch exp/cps3), verify 32 full all-L2 cps 1 (2 consumer WGs/SM) 372.8 vs cps 2 (4 WGs) 190.7 us
  (real 381.8 vs 250.7). So 6 consumer WGs/SM would be ~1.45x faster in all-L2, if they fit and are fed.
- **They don't fit as 3 CTAs/SM**: Triton pads warp-specialized partitions to whole warpgroups (pw 2 -> 12 warps), so
  3 x 384 threads needs maxnreg <= 56 and ptxas rejects it (C7602: the P@V wgmma needs 62). pw 2 / maxnreg 64 runs
  2 CTAs/SM with 18 spills: +199/+146 us.
- Register limits in the current kernel (verify 32 full, real/all-L2 vs 250.7/195): cregs 80 at BN 64: 50 spills,
  +495/+454. pregs 24: +4/+7. BN 32 NS 4: +24/+57; BN 32 cregs 80 pregs 24: +36/+60 (4 spills); BN 32 cregs 72: 20
  spills, +212/+124.
- **One CTA with 3 balanced programs (`_gw_verify3`, plan `v3 1, cps 3, block_n 32, stages 3, maxnreg 72, pregs 24,
  cregs 80`: 6 consumer WGs + producers, 896 threads, 173 KB smem)**: correct (same error as main), but
  - three 1-warp producers (one per ring): no setmaxnreg emitted at all (non-warpgroup partitions), everything at 72
    regs, 20 spills: 449/442 us.
  - one 4-warp producer `_gw_producer3` feeding all three rings per step (flat global-tile loop, request found per tile
    from the balanced index): setmaxnreg 80/24 applied, 2 spills, still 549/496 us (NS 4: 564/508; pregs 40/cregs 72:
    838/674). Not diagnosed. Suspect the single producer warpgroup (gathers + FP8->FP16 V conversion for 6 consumers,
    in-order waits on three rings) or something about 7 partitions in one CTA; ncu warp-state stats would tell.
  Branch exp/cps3 (pushed), tools/s42/regs.py (pass/fail + regs/spills/smem/warps/setmaxnreg for a plan override).

### Next ideas

1. Next UTC day: check main 63a4974 (q16 + kst) against abd67ae2 (decode 1/16/32 full 14.16/90.3/172.8 us).
2. If anyone revisits exp/cps3: ncu `_gw_verify3` (WarpStateStats, per-partition stalls) before changing anything. A
   cheaper test of "is one producer warpgroup enough": in the existing 2-CTA kernel, make the producer do 3x the V
   conversions (redundant) and see whether all-L2 time moves.
3. The 3 us of non-kernel time in check f3e099c1 at decode 16 sliding: look at what differs in launch structure between
   _decode_gluon and the Triton sliding path (grid size 528 vs ?, PDL, _cos_sin), in case it also hides in other shapes.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 43 (overnight agent, 2026-10-24 12:05-~12:45 UTC; unattended session 10 of this night)

**Main 6fb74be = 63a4974 (q16 + kst in `_gd_full`) + batched sliding decode on hpp 2 from batch 24. All UNCHECKED on the
H200** (account checks 0/10 left today; Claude's 3 scored used). Board: #1 2.726x; we're #11 at 2.263x (abd67ae2).
H100 bench 2.033, tests 16/16, tools/s43/scases.py (11 sliding cases incl. batch 24/40, dp, edges, kv_scale) 11/11.

### Per-load view of abd67ae2 (H200, ours/SGLang, per-load speedup)

decode 1 510/1476 (2.89x), decode 16 1916/4436 (2.32x), decode 32 3420/7580 (2.22x), verify 1 606/1625 (2.68x),
verify 16 2455/4607 (1.88x), verify 32 4337/7794 (1.80x). Sliding is 19-25% of the batched loads' time (weight 46)
and 55% of the batch-1 loads'.

### Win (main 6fb74be): hpp 2 for sliding decode at batch >= 24

- New graph-timed gather floor on the real inputs (tools/s43/sfloor.py: bench.time_graph, cold L2, minus empty; each
  program sums its chunk's K and V rows through the page table): decode 32 sliding **10.7 us with whole 128-byte rows,
  12.0 with 64-byte halves**; decode 16 sliding 6.7 / 7.3. Ours: 16.0 / 10.6 (including ~1.6 us of merge launch;
  tools/s43 no-merge ablation: -1.55/-1.60 us).
- So the batched sliding decode plan (hpp 1: each program reads one head's 64-byte half) now uses hpp 2 (block-diagonal
  queries, the whole row) when batch * hkv >= 48: splits 8, block_n 32, 4 warps, 5 stages. Paired H100
  (tools/s36/abc.py, tools/s43/abn.py on nearby batches): decode 32 sliding **-0.66 us (9/9)**, batch 24/40/48/64
  -0.5/-2.4/-3.6/-6.0 (7/7 each); batch 20 +0.2 (kept on hpp 1), decode 16 tie (kept), verify sliding unchanged.
- H200 risk: session 18's hpp 2 for *decode 1 full* won on the H100 and lost on the H200. This one is a bandwidth
  effect that grows with batch, but if the next check shows decode 32 sliding worse than abd67ae2's 14.26 us, revert
  the `heads >= 48` branch in `_plan`.

### Dead ends

- **Triton `kst`** (hpp 2 with heads stacked along the P@V reduction, acc [rows, D], as session 41 did in Gluon):
  splitting the [N, 2D] V tile in registers (reshape/permute/split) **miscompiles at 1-2 warps** (wrong-ish values:
  decode 32 worst row 0.171 vs 0.095; bit-identical at 4 warps). Loading the halves separately is correct but slower:
  decode 32 sliding 16.0 vs 15.1 us (plain hpp 2); hpp 2 + kst for verify sliding 22.8 / 17.4 vs 20.7 / 13.1 (verify
  32 / 16). Code not kept (diff was in /tmp only).
- Decode 32 sliding ablations (paired): constant queries (no qkv/RoPE hop) -0.72 us (an upper bound), no cos/sin load
  -0.26, no new-token work +0.13. Nothing big on the query side.
- **Why the fused last-program merge (plan `fuse`) costs +6 us on batch-1 sliding** (10-11 vs 4.5; the kernel itself
  12.6 vs 4.3 us, tools/s43/prof2.py): not the atomic. Fused with the merge body removed (atomic + reset kept):
  +0.19 us vs the two-kernel path, i.e. a tie. Merge body without the atomic: +5.95. The unrolled
  `static_range(S)` merge serializes into one L2 round trip per split (all 32 [16, 64] FP32 loads at once would need
  ~256 registers/thread). A one-round-trip fused merge would still cost an L2 round trip + compute on top of the tie,
  so it can't beat the PDL merge kernel. Fused merges stay dead; this is the reason.

### Next ideas

1. Next UTC day: check main 6fb74be against abd67ae2: decode 16/32 full 90.3/172.8 (q16 + kst), decode 1 full 14.16,
   decode 32 sliding 14.26 (hpp 2). Revert whichever regresses; submit if the total beats the board.
2. Batched sliding is 1.5-1.8x its gather floor (decode 32: ~14.4 us of attention vs 10.7 floor). The gap is the
   per-tile chain at M = 8-16 rows on Triton's mma-v2 path; Gluon `_gd_full` closes part of it on the H100 but lost
   on the H200 for unexplained cold-start reasons (session 20). Don't retry without an H200 check in the loop.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 44 (overnight agent, 2026-10-24 12:32-~13:05 UTC; unattended session 11 of this night)

**Main d92f1ed+ = 6fb74be + verify sliding from batch 32 on 4 splits x 64-key tiles. All of 63a4974 / 6fb74be / this
is UNCHECKED on the H200** (account checks 0/10 left today; Claude's 3 scored used). H100 bench 2.033 -> 2.057, tests
16/16, tools/s44/scases.py (s43's 11 sliding cases + 5 verify cases at batch 8 dp / 32 / 33 / 64, kv_scale): 16/16,
errors identical to main's to 4 digits.

### Win: verify sliding at batch >= 32 (heads >= 64): splits 4, block_n 64, no peel (was splits 8, block_n 32, peel)

- attn-only vs whole (tools/s38/attn_only.py): attention is a tie (16.5 vs 16.7 us); the gain is the merge, which
  reads half the FP32 partials (whole - attn 3.5 -> 2.3 us). So FP32 partials' cost is recoverable by cutting splits
  where the attention kernel doesn't need them.
- Paired H100 (tools/s36/abc.py, tools/s43/abn.py): verify 32 sliding **-1.57 us (9/9)**; batch 40/48/64 -5.8/-6.7/-7.5
  (7/7); batch 24 -0.16 and 20 +0.88, so the branch starts at heads >= 64. Emulated H200 (tools/emu.py 40): 27.5 -> 21.9.
- H200 risk is low (fewer programs, less traffic), but if the next check shows verify 32 sliding worse than abd67ae2's
  19.05 us, delete the `length > 1 and heads >= 64` branch in `_plan`.

### Dead ends (all verify 32 sliding attn-only/whole sweeps on the new plan, us)

- stages 3/4: +3.5 (!), 6/8 tie; warps 8: +3..+7; maxnreg 128: tie or worse; block_n 128 (splits 2/4): +1..+6;
  splits 2/5: +2..+3; dual (both KV heads as two chains): +4..+14; hpp 2: +18; merge hb 1 +2.2, hb 4 / cw 2: tie.
- Same change for other shapes: decode 32 sliding splits 4 (+0.8..+2.8), decode 16 sliding splits 8 (+0.9), verify 16
  sliding splits 8/12/4, block_n 128, stages 2/3, hb 4: all +0.4..+2.5. Their current plans stay.
- Comment-only cleanup in `_plan`: removed stale "BF16 partials" comments (everything is pdt f32).

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 (H200): decode 16/32 full 90.3/172.8 (q16 + kst),
   decode 1 full 14.16, decode 32 sliding 14.26 (hpp 2), verify 32 sliding 19.05 (this session). Revert whichever
   regresses; submit if the total beats the board (ours 2.263x; #1 2.726x).
2. Merge cost with FP32 partials elsewhere: any shape where whole - attn is large relative to attention and the
   attention kernel ties at fewer splits. verify 16 sliding (whole - attn 2.4 us) didn't work; full layers' merges
   are swept (session 34).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 45 (overnight agent, 2026-10-24 12:38-~13:20 UTC; unattended session 12 of this night)

**Main code unchanged (d92f1ed; all of 63a4974 / 6fb74be / d92f1ed still UNCHECKED on the H200).** 0 judge runs (account
checks 0/10 left today, scored 7/10; Claude's 3 scored used). H100 bench on main 2.056, tests unchanged.

### Measured

- Merge share of the sliding shapes (tools/s38/attn_only.py, whole vs attention only, us): decode 1 4.54/3.04, verify 1
  5.41/3.71, decode 16 10.54/8.96, verify 16 12.85/10.50, decode 32 15.12/13.52. So it's a flat ~1.5 us PDL merge on every
  sliding shape (x46 per load), which sessions 34/36/43 showed can't be fused away.
- **gbal verify-full producer sensitivity**: doing the producer's FP8->FP16 V conversion twice (redundant) costs verify 32
  full +4.5 real / **+26.5 us all-L2** (paired 5/5). lag 2 (convert two tiles behind the gathers, NS 3): +5.7 / +35.6.
  So the producer's per-tile latency sits near the critical path in the H200-like compute-bound regime.

### Dead end: bank-conflict-free V conversion (branch exp/vsw, plan key `vsw`, pushed)

ncu SourceCounters (new tools/s45/conf.py: shared excess wavefronts per SASS instruction) shows the only shared-memory
conflicts in `_gw_verify` (gbal) are the producer's 8 `LDS.64` V loads (unswizzled ring, 4 rows per warp on the same
banks: exactly 2x, 2.59M excess wavefronts at verify 32 full). Fix `vsw 3`: V ring in the 128B-swizzled NVMMA layout and a
DistributedLinearLayout with 16 B per thread where each 8-lane LDS/STS.128 phase covers rows r and r^5 (disjoint
swizzle chunks for both the FP8 ring and the FP16 tiles): **0 excess wavefronts**, 2% fewer instructions, bit-identical,
but paired verify 16/32 full real -0.19/+0.26, all-L2 +0.58/+1.15 us. (vsw 1/2, plain BlockedLayouts on the swizzled
ring: still 2x conflicts, since a 128-bit phase is 8 lanes = rows r, r+1, which share chunks; vsw 1 also conflicts on the
stores, +10 all-L2.) The producer's cost is latency (top stall: mbarrier spin `BRA long_sb`), not LDS throughput.
- Also dead: skipping the redundant window mask on whole tiles for single-token sliding decode (decode 1/16/32 sliding
  +0.26/+0.08/0.00 us paired 9 rounds).

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 44's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05); revert whichever regresses, submit if the total
   beats the board.
2. gbal producer latency: the conversion's critical path (wait_group -> barrier -> LDS -> F2FP -> STS -> fence -> arrive)
   with only NS 3 slots. A deeper ring doesn't fit (2 CTAs x 96 KB). Possible lever: decouple the FP16 V tiles from the
   FP8 ring (own 2-slot ring + barriers) so the FP8 ring can be 4-5 deep in the same smem. Untested.
3. Tools: tools/s45/abl2.py (paired A/B of two submission files at decode/verify full, real + all-L2), conf.py.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 46 (overnight agent, 2026-10-24 12:48-~13:10 UTC; unattended session 13 of this night)

**Main code unchanged (d92f1ed; 63a4974 / 6fb74be / d92f1ed still UNCHECKED on the H200).** 0 judge runs (account checks
0/10 left today, scored 7/10; Claude's 3 scored used).

### Dead end: FP16 V tiles on their own ring in gbal (branch exp/dvr, plan key `nv`, pushed)

Session 45's idea 2. With `nv` > 0, `_gw_verify`'s BAL path keeps the FP16 V tiles in an NV-slot ring with its own
`vempty` barriers (consumers arrive after P@V; the producer waits on it before converting), and `empty` frees an FP8 K/V
slot as soon as QK^T has read K (its V bytes were already converted by the producer). So the FP8 ring can be deeper in
the same 96 KB. Bit-identical to main (tools/s46/ident.py). Paired verify 32 full, 5 rounds (tools/s46/psweep.py), us:

| plan | real | all-L2 |
|---|---|---|
| nv 2, stages 4 (96 KB) | +0.3 | **+7.2** |
| nv 2, stages 4, lag 2 | +0.9 | +3.1 |
| nv 3, stages 3 (same layout as main + early K free) | +0.2 | +0.7 |
| nv 2, stages 5 (112 KB: 1 CTA/SM) | +28 | +48 |

- A deeper FP8 ring buys nothing on real time (so the gathers aren't short of in-flight depth), and a 2-slot FP16 ring
  puts the producer's conversion latency on the consumers' path (H200-like all-L2 +7). That agrees with session 45:
  the producer's per-tile latency is what matters, and the ring depth doesn't help it.
- The first version arrived on `empty` right after the QK^T wait: nv 3/ns 3 was +7.3 all-L2. Moving that arrive to just
  after issuing the async P@V brought it to +0.7. **An mbarrier.arrive in a consumer partition costs real latency (it
  syncs the partition) when it sits on the QK^T -> softmax chain**; put arrives where they overlap an async wgmma.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 44's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05); revert whichever regresses, submit if the
   total beats the board.
2. gbal smem layout is exhausted at 2 CTAs x 96 KB (ring depth, V conversion layout, decoupling: all tie or lose).
   What's left there is cutting the producer's per-tile chain itself (fewer instructions between wait_group and the
   ready arrive), e.g. converting in two halves with an arrive per consumer head (vh0 ready before vh1 is converted).
   Untested.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 47 (overnight agent, 2026-10-24 13:07-~13:55 UTC; unattended session 14 of this night)

**Main code unchanged (d92f1ed; 63a4974 / 6fb74be / d92f1ed still UNCHECKED on the H200).** 0 judge runs (account checks
0/10 left today, scored 7/10; Claude's 3 scored used). H100 bench on main 2.038 at session start (other agent timing too;
per-shape numbers within noise of session 44's).

### Dead end: K-ready barrier in gbal (branch exp/kr, plan key `kr`, pushed; all bit-identical)

Idea: a second per-slot barrier `kready` that the producer arrives right after `wait_group` + `gl.barrier` (+ fence),
so the consumers run QK^T and the softmax while the producer converts V, and wait on `ready` only before P@V. Paired
verify full (tools/s46/psweep.py), us:

| kr | what | B16 real / all-L2 | B32 real / all-L2 |
|---|---|---|---|
| 1 | K-early (fence + kready arrive, ready wait before P@V) | +0.4 / +6.0 | +0.8 / +12.5 |
| 2 | overhead only (both waits at the loop top, extra fence + arrive in producer) | | +0.4 / +2.1 |
| 3 | K-early without the producer's extra fence (racy, diagnostic) | | +0.5 / +19.6 |
| 4 | tail mask applied on every tile (no branch) | +1.6 / +12.4 | +4.2 / +25.4 |
| 5 | tail mask under `i == ntiles - 1` instead of `lo + (i+1)*BN > hi` | | +1.4 / +9.7 |
| 6 | no tail mask at all (wrong, diagnostic) | | -0.8 / 0.0 |

- An mbarrier wait between the softmax and P@V costs far more than the conversion overlap buys (kr 2 vs 3), same
  lesson as session 46 for arrives: keep consumer barrier ops off the QK^T -> softmax -> P@V chain.
- kr 5 vs main is the same work under a different scalar condition and costs +9.7 us all-L2; kr 6 shows the
  current tail branch costs nothing. So **`_gw_consumer_bal` timing swings ~5% with ptxas scheduling of
  semantically equal code**; read single-variant A/Bs on this kernel with that in mind (kr 4's +25 doesn't prove
  issue-boundness).

### Idea not tried (needs care with the rules)

- FA4-style speculative softmax: compute p against the previous tile's max so the row-max reduction (5 FMNMX + 2
  shuffles) leaves the chain, with an exact fix-up when the max grows. Needs a warpgroup-uniform decision before any
  wgmma (warp-level votes would diverge the warpgroup) and a data-dependent fallback branch, which sits uneasily with
  CLAUDE.md's "decisions only on shapes". Session 26's wcr (warp vote + skip) was dead. Don't start without a design
  that keeps P@V unconditional.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 44's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05); revert whichever regresses, submit if the
   total beats the board.
2. The gbal consumer chain is out of Triton/Gluon-level levers (sessions 35-47). The measured lever remains 6 consumer
   WGs/SM (session 42), which is native code (Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 48 (overnight agent, 2026-10-24 13:13-~13:25 UTC; unattended session 15 of this night)

**Main 652aafe = d92f1ed + decode 16 sliding on hpp 2 / 64-key tiles. 63a4974 / 6fb74be / d92f1ed / 652aafe are all
UNCHECKED on the H200** (account checks 0/10 left today; Claude's 3 scored used). H100 bench 2.035 (session start, other
agent timing) -> 2.055; tests 16/16; tools/s43/scases.py 11/11; new tools/s48/cases.py (6 decode cases that hit the new
branch: batch 16 tp8 page 64 / kv_scale / ctx 2047 / edges, dp batch 2, dp batch 4) 6/6.

### Win: decode 16 sliding (heads == 32, length 1): hpp 2, splits 16, block_n 64, 4 warps, 3 stages

- Was hpp 1, block_n 64, 3 stages (the `heads <= 32` branch). Session 43 found hpp 2 a tie at decode 16 with that
  branch's block_n 32 / 5 stages; with two whole 64-key tiles per split it wins: paired H100 (tools/s36/abc.py)
  **-0.14 us 13/15 (A/B) and +0.16 3/15 (B/A)**; bench 10.62 -> 10.27 us.
- Same block_n 64 / 3 stages at decode 32 (8 splits): +0.14 (3/11), so decode 32 keeps block_n 32 / 5 stages.
- Also hits dp batch 2 (heads 32), which was on the hpp 1 plan; correct, not timed.
- H200 risk: same kind as session 43's hpp 2 (whole 128-byte rows). If the next check shows decode 16 sliding worse
  than abd67ae2's (judge) time, drop the `heads == 32` branch in `_plan`.

### Dead ends (sweeps + paired, us)

- Decode 32 sliding warps 2: sweep -0.3, paired +0.14 (2/11) / 0.00 (5/11): tie. stages 8, splits 6/12: +0.1..+1.
- Verify 16 sliding: stages 8, block_n 16, warps 2, splits 16, hpp 2: all +1.5..+22.
- Verify 1 full: splits 191/255, block_n 32 / 8 stages, 8 warps: all +1.8..+5.6 (current plan best).

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 44's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05) plus decode 16 sliding (this session); revert
   whichever regresses, submit if the total beats the board.
2. The Triton plan space on the sliding shapes is close to exhausted (sessions 26/35/43/44/48); remaining gains there
   are in the per-tile chain at M = 8-16 rows (mma-v2), i.e. native code (Codex's lane) or a Gluon path that survives an
   H200 check.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 49 (overnight agent, 2026-10-24 13:32-~14:10 UTC; unattended session 16 of this night)

**Main code unchanged (652aafe; 63a4974 / 6fb74be / d92f1ed / 652aafe all still UNCHECKED on the H200).** 0 judge runs
(account checks 0/10 left today; Claude's 3 scored used). H100 bench on main at start 2.035 (same per-shape numbers as
session 48). Board moved: #1 2.726x, we are #13 at 2.263x.

### Dead end: FP16-pair exp in the gbal consumer (branch exp/hex2, plan key `hx`, pushed)

ncu op mix of `_gw_verify` at verify 32 full, all-L2 (tools/s40/one_l2.py + ops.py): 1.20e8 warp instructions, ~375 per
consumer warp-tile (FFMA/FMNMX/FMUL/MUFU/F2FP/FADD ~9% each, BRA spin 7.7%, SYNCS 6.9%); ~65% issue utilization.
- `hx 2`: P = ex2.approx.f16x2 of the FP32 exponent rounded to FP16, row sums as P @ ones (FP32 wgmma). Accuracy same
  as main (verify 16/32 full rel err 0.0304/0.0313 vs 0.0304/0.0313), but paired 7 rounds real +0.66/+0.45, all-L2
  **+3.5/+6.9 us**. Reason in SASS: **sm_90 has no packed f16x2 MUFU**: ex2.f16x2 becomes two MUFU.EX2.F16 + PRMTs
  (PRMT 4.4% of instructions), so the MUFU count doesn't drop and the packing is extra. Don't retry on Hopper (it's a
  Blackwell trick).
- `hx 1` (P @ ones row sums only, FP32 exp): tie (real -0.22/+0.13, all-L2 +0.18/+0.45), as session 35's exa 5.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27); revert whichever
   regresses, submit if the total beats the board.
2. Verify full (the weakest loads, 1.74-1.79x on the H200) is out of Triton/Gluon-level levers; FP8 P@V (needs an FP8
   V transpose) and 6 consumer WGs/SM are native code (Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 50 (overnight agent, 2026-10-24 13:39-~14:10 UTC; unattended session 17 of this night)

**Main code unchanged (652aafe; 63a4974 / 6fb74be / d92f1ed / 652aafe all still UNCHECKED on the H200).** 0 judge runs
(account checks 0/10 left today; Claude's 3 scored used). H100 bench on main at start 2.048 (per-shape numbers as
session 48).

### Dead end: one-pass fused last-program merge (branch exp/fm, plan key `fm` = rows per chunk, pushed)

Why session 36's fused merge was 10-11 us at batch-1 sliding (vs 4.5): the old `FUSE` tail in `_attn_split` unrolls
`static_range(S)` per-split loads at 96 regs, and ptxas runs them as ~32 serial L2 round trips (no spills; ncu
`_attn_split` 14.3 us fused vs 5.3 + 3.9 separate). `_fused_merge` loads all slots' [RB, D] blocks in one 3-D load
(layout keeps the split dim in-thread, so the reduction needs no smem), with the gate preloaded before the arrival.
H100 sweeps, us (base / fused):
- decode 1 sliding 4.48 / 5.6 (fm 8; tsep kept). Ablation copies: acq_rel arrival, no merge body 4.42; relaxed
  arrival, no merge 4.06. So **the acq_rel arrival (release fence) alone costs about what the whole PDL merge kernel
  costs**, and the merge body adds ~1.2 us on top. A fused merge can't beat PDL at batch 1. Don't retry.
- verify 1 sliding 5.57 / 16.7 (row chunks run serially); decode 16/32 sliding 10.42/14.88 / 11.54/16.32; verify 16
  12.67 / 18.05; verify 32 18.83 / 21.6. All worse.
- Also re-checked verify 1 full: dual 21.9, dual + splits 191/255 22.9-23.8, hpp 2 / 255 splits 41.7 vs 18.2 now.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27); revert whichever
   regresses, submit if the total beats the board.
2. Merge-kernel removal is closed on every path (PDL, flags, clusters, fused last-program). Verify full remains the
   weakest load and is native-code territory (Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 51 (overnight agent, 2026-10-24 13:46-~14:40 UTC; unattended session 18 of this night)

**Main code unchanged (652aafe; 63a4974 / 6fb74be / d92f1ed / 652aafe all still UNCHECKED on the H200).** 0 judge runs
(account checks 0/10 left today; Claude's 3 scored used). H100 bench on main at start 2.04 (decode 16 sliding 10.42,
others as session 48).

### Dead end: 6 consumer warpgroups per SM again, with consumers converting their own V (branch exp/v3cv, plan key `v3n`, pushed)

Session 42's exp/cps3 (3 balanced programs per CTA) was 549/496 us with one producer warpgroup that both gathered and
converted V for all six consumers. `_gw_verify3n` instead has a gather-only producer (`cp.mbarrier_arrive` on `ready`, as
`_gw_producer`); each consumer converts its own head's V half while its QK^T wgmma runs (as `_gw_consumer`), then
arrives on `empty` after issuing P@V. BN 32, NS 4, maxnreg 72 (896 threads), cregs 80. It's correct (verify 32 full rel err
0.029). Verify 32 full, real/all-L2 in us (main ~256/200):

| variant | real | all-L2 |
|---|---|---|
| lsum row sums, pregs 32 (26 spills) | 448 | 391 |
| no lsum (`lsum3` false), pregs 24 | 439 | 368 |
| + next step's page indices loaded before the slot waits (3 rings) | 369 | 344 |
| pregs 32 | 321 | 306 |
| producer as the default partition (`pdef`), pregs 24 | 352 | 313 |
| `abl3 1` (no page loads in the loop, wrong results) | 332 | 333 |

- **Register gotcha**: in Gluon's `warp_specialize` the default partition (first entry) gets what's left of the budget, and
  that can be less than the worker regs: with workers 5x80 + 32 the default consumer ran at 72 and spilled its accumulator
  around every QK^T (STL/LDL in its loop only). With pregs 24 it got 80 and the 24-register producer spilled instead.
- ncu (all-L2, pdef): **190M instructions vs main's ~120M**, at the same ~65% issue rate. BN-32 per-tile overhead, the
  producer's per-tile `_gw_tile` reductions and the mbarrier spins of 28 warps all cost issue slots. Session 42's linear
  scaling (2 -> 4 WGs) stops at 4 WGs because the SM is already near its issue limit. Don't retry 6 WGs/SM in Gluon.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27). Revert any shape
   that got worse, and submit if the total beats the board.
2. Verify full on the H200 is issue-bound (~65% at 4 WGs/SM), so what's left is fewer instructions per key in the
   consumer, which sessions 35-49 have exhausted at the Gluon level.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 52 (overnight agent, 2026-10-24 13:55-~14:45 UTC; unattended session 19 of this night)

**Main bf5123e = 652aafe + `dbal` for decode full from heads 48 (batch >= 24 at tp8). 63a4974 / 6fb74be / d92f1ed /
652aafe / bf5123e all UNCHECKED on the H200.** 0 judge runs (account checks 0/10 left today; Claude's 3 scored used).
H100 bench 2.043 at start, 2.035 after (noise; decode 32 full 241.2 -> 240.4). Tests 16/16, tools/s52/cases.py 8/8
(new: short/mixed/edges/kv_scale/batch 128 cases that reach dbal), tools/s19/cases.py 9/9.

### Win (H200-leaning): length-proportional splits for the Gluon decode-full kernel (plan key `dbal`, session 21's idea 3)

- `_gd_full` with `BAL`: grid (P, 1, hkv/2), P = 2048 / hkv programs per KV head pair. Request b owns program ids
  [c_b, c_b+1), c_b = b + (P - B) * excl_tiles_b // T_tiles (>= 1 each; no program spans requests, so no extra pipeline
  prologues, unlike gbal's segments). Partials by id in `_combine_bal`'s layout (key partial at p + b, new token at
  P + B + b); `_combine_bal` got `PROP` (exact [c_b, c_b+1) range instead of gbal's predicate).
- H100 paired (tools/s52/ab.py = plan A/B real + all-L2 on bench data; tools/s52/near.py = nearby batches, seed 7):
  decode 32 full (bench data) real +0.4, all-L2 -4.9 us; seed 7 decode 24/32/48/64 real -3.5/-3.5/-3.7/+0.5, all-L2
  -6.6/-7.5/-16.6/-14.5. Batch 4/8/12/16 lose (real +0.8..+3.2), so it's gated at heads >= 48 (decode 16 unchanged).
- Merge matters: `_combine_bal` with 2 heads x 32 slots per one-warp program cost +1.3 us of merge tail at decode 32
  (requests get 31-35 programs: two passes); 1 head x 64 slots, 2 warps (hb 1, sb 64, cw 2) is +0.2. Program count
  matters too: dprogs 960 loses, 1056 is far worse (+26..+60, residency with the PDL merge), 1024 best.
- Real-mode A/Bs on the bench's data swing ~1.5 us between runs on this shape (BW-bound on the H100); trust all-L2
  plus both-order paired runs. tools/s36/abc.py runs A then B in fixed order: run it both ways.
- H200 expectation: decode 32 full -1..-3 us of 172.8 (~0.1-0.2% overall). If the next check shows decode 32 full
  worse than abd67ae2's 172.8, drop the `heads >= 48` dbal branch in `_plan` first.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27); revert any shape
   that got worse (decode 32 full: dbal first, then kst/q16), submit if the total beats the board.
2. decode 16 full: dbal loses on the H100 at batch <= 16 (real +0.9, all-L2 -0.9..-1.8). The merge for ~64 slots per
   request may be the cost; a 2-pass-free layout (sb 64/128 with cw 4) was tried at 1008/1024 programs without a clear
   real win. Low odds.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 53 (overnight agent, 2026-10-24 14:05-14:15 UTC; unattended session 20 of this night)

**Main fa52b18 = bf5123e + no window mask on decode's cached keys. 63a4974 / 6fb74be / d92f1ed / 652aafe / bf5123e /
fa52b18 all UNCHECKED on the H200** (account checks 0/10 left today; Claude's 3 scored used). H100 bench 2.045 at start ->
2.05; tests 16/16; outputs + caches bit-identical to main on 10 decode sliding cases (/tmp-only script, pattern: run
base and new kernel_fn on the same reference.generate_input data, restore caches in between, torch.equal).

### Where the batched sliding time goes (H100, attention only via tools/s38/attn_only.py)

verify 32 / verify 16 / decode 32 sliding: whole 18.7 / 12.9 / 15.1 us, attention 16.4 / 10.6 / 13.6 (merge ~1.5-2.3).
Ablation copies (paired, tools/s36/abc.py, 7 rounds):
- split 0 without the block's new tokens (wrong results): -0.24 / -0.27 / +0.13. **Split 0's new-token work is not
  the critical path** (so uneven chunking that gives split 0 less work can't pay).
- page index replaced by a synthetic strided row (`t*7 % 60000`, no page-table load): -2.8 / -1.3 / -1.5. But the same
  synthetic rows *with* the page-table load kept as a dependency (`(load >> 30) + t*7 % 60000`) are equal (-0.16 /
  +0.02 / +0.29): **the page-index chain costs nothing; the 1.3-2.8 us is the random-row gather pattern itself**.
  (Gotcha: `load * 0 + x` gets folded and drops the load; use `>> 30`.)

### Win (small): decode sliding without the window mask on cached keys

Cached keys start at base = max(seq - 2047, 0) and a decode query sits at seq, so `t > pos - WINDOW` always holds for
them; `_attn_split` now passes WIN = 0 to `_tile` when L == 1 (the new token's own path is unchanged). Paired H100,
15 rounds both orders: decode 16 sliding -0.14 (15/15) / -0.14 (11/15), decode 1 sliding -0.05 / 0.00, decode 32
-0.02 / -0.06. H200 risk: none expected (fewer instructions, identical results).

### Dead end: verify sliding window-mask elision (not committed; described here)

A verify row j masks only keys below base + j, i.e. split 0's first tile. Tried (plan key `we` on exp/wmask, then
reverted): (a) that tile in its own one-iteration loop before the main loop: verify 16/32 sliding +1.4/+1.7 us (split 0
runs one unpipelined tile, and split 0 sets the critical path); (b) that tile as iteration 0 of the masked tail loop
(`start = where(i < first, lo, ...)`), main loop unmasked: needs `peel` for any gain, and peel at 64 rows costs
registers: verify 16/32 +0.35/+0.16 without peel, +1.5/+1.0 with; verify 1 sliding (one tile per split) +peel
-0.14 in a sweep but a tie paired (+0.06 / 0.00). Also re-swept `dual` for verify 16/32 sliding: +3.5..+7 us.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27). Revert any shape
   that got worse (decode 32 full: dbal first), submit if the total beats the board.
2. Batched verify sliding (weight 46, ~1.8-2.3x on the H200) is per-tile-chain bound at 2 WGs/SM like verify full; the
   random-row gather pattern is worth ~2.8 us at verify 32 but isn't controllable.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored (per this
morning's count; `okc problems` shows 0 checks left).

## Session 54 (overnight agent, 2026-10-24 14:14-~14:30 UTC; unattended session 21 of this night)

**Main code unchanged (fa52b18; 63a4974 / 6fb74be / d92f1ed / 652aafe / bf5123e / fa52b18 all still UNCHECKED on the
H200).** 0 judge runs (`okc problems`: checks 0/10 left today, scored 3/10 left but no check to qualify a submit).
H100 bench on main at start 2.045 (per-shape as session 53); tests 16/16 after the session.

### Dead ends (gbal verify-full kernel, `_gw_verify` with BAL; new tools/s54/sass.py = SASS + op counts of
`_gw_verify` at a verify-full batch, tools/s54/abl2.py = paired A/B of two submission files, alternating order, real
and all-L2)

- **V FP8->FP16 by integer bit tricks** (e4m3 byte -> f16 bits `(b&0x80)<<8 | (b&0x7f)<<7` = value x 2^-8, exact,
  the 2^8 folded into the scale; columns may be permuted freely since acc columns are only a store index). Not built:
  the producer's SASS already does one `F2FP.F16.E4M3.UNPACK_B` per 2 elements (32 per thread per tile, no PRMT), and
  the shift/mask form needs 3-4 ALU ops per 2 elements. The only case it wins is if the F2FP pipe were the limit, and
  session 35's `exa 1` (no exp: -3% all-L2) shows the MUFU/F2FP-class pipes aren't.
- **Ring slot/phase carried across tiles instead of `(it+i) % NS`, `((it+i)//NS) & 1`** (NS = 3, so a mul-hi
  chain on the uniform datapath right before the `ready` wait; ~4 uniform ops less per tile). Consumer only: paired 9
  rounds verify 32 full real +0.03, all-L2 -0.13 (7/9); verify 16 real -0.02, all-L2 -0.13 (7/9): noise-level, not
  kept. Producer too (its `empty` wait index and the lagged convert index carried with scf.if wraps): all-L2 +0.88
  (0/9) / +0.26 (1/9), worse.
- **TMA / cp.async.bulk row copies for the producer's gathers** (would cut the ~9 us all-L2 the gather issue costs,
  session 35 `exa 3/4`). Not built: k_ring is 128B-swizzled for the QK^T wgmma and a linear 128-byte bulk copy can't
  swizzle; tensor TMA (box [1, 128]) is issued from one thread with scalar coordinates in Gluon, i.e. 64 scalar page
  lookups per tile. (Hopper has no TMA gather4; FA3/FlashInfer use cp.async for page size 1 for this reason.)

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27). Revert any shape
   that got worse (decode 32 full: dbal first), submit if the total beats the board (2.263x ours; #1 2.726x).
2. The Triton/Gluon lane looks exhausted on the H100 (sessions 35-54). Verify full (1.74-1.79x on the H200, the
   weakest loads) is issue-bound in the consumer at 4 WGs/SM; what's left there is native code (Codex's lane).

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 55 (overnight agent, 2026-10-24 14:22-~14:45 UTC; unattended session 22 of this night)

**Main 1114bf9 = fa52b18 + `pre` (verify 1 full only). 63a4974 / 6fb74be / d92f1ed / 652aafe / bf5123e / fa52b18 /
1114bf9 all UNCHECKED on the H200** (`okc problems`: checks 0/10 left today; scored 3/10 left but nothing qualifies a
submit without a check). Tests 16/16; H100 bench 2.042 at start, 2.041 after (noise; verify 1 full 18.29 -> 17.98).

### Win: L2 prefetch of the split's first tiles before the query norm (plan key `pre`, `_attn_split`)

- With `PRE = n`, `_attn_split` computes its key range first (helper `_range`), loads the page indices of its first
  n tiles and issues `prefetch.global.L2` for their K and V rows (inline asm, one row per element), then does the
  query load/norm/RoPE as before; the pipelined loop's cp.async gathers then hit L2. So the page-table -> gather chain
  overlaps the qkv (and cos/sin) chain instead of following it.
- Paired H100 (new tools/s55/abalt.py: two submission files, 20 rounds, order alternating each round): verify 1 full
  pre 1 -0.45 us (20/20), pre 2 another -0.48 (19/20), pre 4 +1.06 vs pre 2. Final vs main: **verify 1 full -0.78
  us (20/20, -0.79 / -0.77 by order)**. Nearby (tools/s55/near.py, seed 7, all correct): B1 ctx 40K/120K/300K
  -1.10/-0.77/-0.33, B1 page 64 -0.28, B2 ctx 80K -0.79.
- Expected H200: verify 1 full ~-0.5..-0.8 us of 17.18 (batch-1 full ~0.5% overall per us, so ~+0.3%). It's latency
  overlap, not a parallelism change, so lower risk than the session 18 plan flips; if the next check shows verify 1
  full worse than abd67ae2's 17.18, set `pre=2` off in `_plan`'s verify-1-full branch first.
- Gotcha: just moving the `base/lo/hi` computation above the query load (no prefetch) cost decode 1 sliding +0.22 us
  (0/20). The non-PRE path keeps it after the queries.

### Dead ends (measured, not kept)

- First tile held in registers (loaded before the queries, consumed before the loop; loop from lo + BLOCK_N), for
  all `_attn_split` plans: decode 1/16/32 sliding +1.1/+2.6/+2.0 us, verify 16/32 sliding +0.6/+0.9, verify 1 full
  +1.1; verify 1 sliding -0.06 paired (13/20). Register-held tiles replace the pipeliner's cp.async path.
- `pre` (L2 prefetch) on sliding: batch 1 ties (one 64-key tile per split, nothing beyond it to prefetch); decode
  16/32 and verify 16/32 sliding +0.2..+0.7 us.
- The same L2 prefetch in Gluon `_gd_full` for decode 1 full (rows of the PFT tiles after the NS - 1 cp.async
  prologue tiles, one row per lane): pft 1/2/4 +0.6..+0.8 us. `_gd_full` already issues its prologue cp.async before
  the query math.
- Verify 1 full with pre 2: stages 10 -0.14 us paired (15/20), within judge noise, not kept; stages 6/7, splits
  63/95, block_n 32 not better in a sweep.

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27) plus verify 1 full
   17.18 (this session). Revert any shape that got worse (decode 32 full: dbal first), submit if the total beats the
   board (2.263x ours; #1 2.726x).
2. Other latency-bound prologues that wait on a dependent chain: none left in Triton (batched sliding loses, batch-1
   sliding has one tile per split). The merge kernels already load the gate before the PDL wait.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.

## Session 56 (overnight agent, 2026-10-24 14:32-~14:50 UTC; unattended session 23 of this night)

**Main code unchanged (1114bf9; 63a4974 / 6fb74be / d92f1ed / 652aafe / bf5123e / fa52b18 / 1114bf9 all still
UNCHECKED on the H200).** 0 judge runs (`okc problems`: checks 0/10 left today, scored 3/10 left but nothing qualifies a
submit without a check). H100 bench on main at start 2.043 (decode 1/16/32 sliding 4.59/10.30/15.60, full
15.50/129.01/240.77; verify 1/16/32 sliding 5.74/13.17/19.47, full 17.65/136.11/250.85).

### Dead end: BN-32 tiles for the gbal verify-full consumer (prerequisite for session 35's idea "QK(i+1) during softmax(i)")

The only untried item left in the notes' lists was BN 32 tiles with QK^T of tile i+1 issued during softmax(i) (two
16-register S tiles in the budget of one BN-64 tile). Measured the unpipelined BN 32 plan first (only `block_n=32` in
the Gluon verify branch; tools/s54/abl2.py, 3 rounds paired): verify 16 full real +11.1 / all-L2 +22.5 us, verify 32
full real +26.1 / all-L2 **+46.1 (198.7 -> 243.6, +23%)**. The per-tile overhead (ready/empty waits, wgmma waits,
reductions per tile) doubles, so the overlap would have to win back >23% of compute time just to tie. Session 14's
BN-64 FA3 pipelining already lost (+6%). Not built. **Don't retry BN < 64 on the gbal consumer.**

### Next ideas

1. Next UTC day (checks reset 00:00Z): check main against abd67ae2 per session 48's list (decode 16/32 full 90.3/172.8,
   decode 1 full 14.16, decode 32 sliding 14.26, verify 32 sliding 19.05, decode 16 sliding 10.27) plus verify 1 full
   17.18. Revert any shape that got worse (decode 32 full: dbal first; verify 1 full: pre), and submit if the total
   beats the board (2.263x ours; #1 2.726x).
2. Every idea listed in sessions 14-55 for the Triton/Gluon lane has now been measured. Until a check shows the H200
   numbers for the 7 unchecked commits, more H100 micro-tuning is unlikely to pay. The next sessions should be short:
   just the check in idea 1.

### Budget (UTC day 2026-10-24)

This session: 0 checks, 0 scored. Claude day: 5/8 checks, 3/3 scored. Account: 10/10 checks, 7/10 scored.
