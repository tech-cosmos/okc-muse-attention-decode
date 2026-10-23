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

UTC day 2026-10-23: 1 check used (6ffea83d), 0 submissions.
