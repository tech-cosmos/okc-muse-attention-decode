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
