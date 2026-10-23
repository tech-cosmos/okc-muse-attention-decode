# Codex notes: muse_attention_decode

Newest entries at the bottom. Codex owns native CUDA / CuTe / inline-PTX kernels;
Claude owns the Triton implementation. Read CODEX.md, CLAUDE.md, TASK.md, AGENTS.md,
NOTES.md, and ~/COORD.md before the next cycle.

## Session 1 — 2026-10-23, 20:24–20:54 UTC

### Outcome and branch state

No native speedup to dispatch this cycle. Built and measured 15 native verify-full
variants, preserved the strongest working prototype, and kept submission.py on the
latest inherited Triton implementation. No judge checks or scored runs were used.

- Experiment: `codex-exp/cuda-verify`, pushed through **42f8356** (kernel/results
  commit **6b3f03d**). Start with `experiments/README.txt` and
  `experiments/cuda_verify.py`. Historical source variants and measurements are
  retained there; they are deliberately not imported by submission.py.
- Production branch: `codex`, fast-forwarded from 8743290 to **origin/main 0e5aa7a**
  at session end. This includes Claude's eedd003 register-cap/wave-fill change.
  submission.py is byte-for-byte identical to origin/main, with no native dispatch.
- NOTES-codex.md did not previously exist; this is its first entry.
- Neither ~/STOP nor ~/STOP-codex existed during the cycle.

### Measurements (H100, cold-L2 CUDA graphs)

Initial inherited baseline reproduced at 1.800x against the harness's fixed
SGLang-on-H200 numbers: verify 16/32 full = **159.86 / 305.38 us**.

Final native comparison, same process and graph timing method, before the main merge:

| shape | native | inherited Triton |
|---|---:|---:|
| verify 1 full | 34.13 us | 18.83 us |
| verify 16 full | 235.97 us | 161.09 us |
| verify 32 full | 449.06 us | 304.90 us |

The final sweep minima were about 234 / 446 us, so the retained native kernel is
roughly 47% slower on the two target shapes. The existing implementation stays in use.
All timing loops use tools/bench.py's GPU flock, cold-L2 flush, graph replay, and
empty-replay subtraction. These are local H100 results, not H200 judge measurements.

After merging the new main, `python tools/bench.py test` passed 16/16, and `bench`
reported **1.794x**, verify 16/32 full **155.89 / 306.16 us**. The aggregate includes
some batch-1 timing variation (decode-1 full was 18.53 vs 17.31 us initially); no
extra tuning or repeated measurement was done to select a flattering aggregate.
Complete final main validation is in `tools/codex-session1-validation.txt`.

### Native design and lessons

The retained native kernel supports full attention, L=8, D=64, G=8, pages 1/64,
and arbitrary batch/KV-head counts within that layout. One CTA has one producer
warp and four consumer warps. It balances context splits per request on device.
The producer issues cp.async gathers into a shared-memory ring; consumers use
FP8 WGMMA for QK and FP16 inline-PTX MMA for PV. Normalization, FP8 query rounding,
new K/V writes, causal new-token attention, and every cached key are computed
inside the timed call. Partials remain FP32. The existing Triton _combine handles
the split reduction and output gate. No input-derived state is cached.

Useful implementation findings:

- Scalar FP8 conversion and signed indexing cost far more than expected. Packed
  fp8x2->half2 conversion, unsigned constant indexing, and padded shared layouts
  reduced the initial 1590 / 3160 us WMMA prototype to about 321 / 612 us after
  choosing suitable split counts.
- On this H100, the generated SASS for warp-level `mma.sync` FP8 included FP8->FP16
  conversion and HMMA.16816 instructions. Switching QK to Hopper's true FP8
  `wgmma.mma_async` reduced roughly 295 / 578 to 255 / 490 us. Target sm_90a.
- K uses a no-swizzle 8-row x 16-byte blocked shared layout for WGMMA. Its descriptor
  encodes leading byte offset 128 and stride byte offset 512. The producer scatters
  each row's four 16-byte pieces directly into this layout.
- Q and all accumulated/output rows retain the same logical 64-row grouping as the
  Triton partial layout. Native pipeline flags use release/acquire ordering, named
  consumer barriers, and async-proxy fences for ordinary shared stores. The retained
  code also explicitly fences compiler access to asynchronous accumulator registers.
- Moving the scale into row maxima and exp2 FMA, masking only the final cached/new
  tiles, and skipping multiplication by exactly 1 when maxima do not change gave
  the final ~234 / 446 us. This still computes every key and value; it does not skip
  attention work based on weights.
- The final module builds at import into experiments/.build, explicitly targets
  sm_90a, and locally compiles comfortably within the judge's 15-minute allowance. CUDA
  uses light ATen/c10 headers to avoid repeatedly compiling torch/extension.h.

Dead ends to avoid repeating without a new structural idea:

- FP16 WGMMA PV: 289 / 565 us; overlapping it with the next tile improved only to
  281 / 541. Warp MMA PV with register-packed probabilities was better here.
- 64-key tiles: best ~244 / 460, 128 registers and some spilling. In the archived
  v11 driver, stage label 4 actually aliases stage 3 to respect static shared limits.
- Cooperative loads by all four compute warps, instead of a dedicated producer:
  ~264 / 510; no gain from eliminating producer polling.
- Ping-pong next-tile QK overlapped with current softmax/PV: ~267 / 502. Register
  usage increased to 124, reducing residency to three CTAs/SM.
- Gather depths 5, 6, 7: no improvement over 3/4. Every swept plan's output was
  compared to the reference-validated default. This was checked after reading
  Claude's new pipeline-depth observation; the native design did not share the win.
- Do not infer clean timing from v2: its initial bench overlapped an Nsight run.
  That data is marked invalid in session1-measurements.json. All later profiler
  runs took /tmp/okc-gpu.lock. The first v1-named Nsight file actually profiled v2
  because the module import occurred after an edit; the text export is named v2.

PTX layout documentation consulted:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions

### Validation

- Native/fallback dispatch passed the full 16-case local reference suite. Five
  full-verify cases use native code; the remaining cases retain the existing kernel.
- Seven additional native cases passed: batches 1,2,4,8,12,24,48; small and long
  contexts; pages 1/64; normal/soft/coherent/edge data; 1/64 cache scale.
- The three scored full-verify cases passed repeatedly during tuning.
- Compute Sanitizer memcheck and synccheck each reported **0 errors** on six small
  native cases, pages 1/64, scales 1 and 1/64, contexts 10/257/1024, depths 3/5/7.
- Latest-main production kernel separately passed 16/16 after the final merge.
- Native results and sanitizer logs live on codex-exp/cuda-verify under
  experiments/results; reproducibility instructions are in experiments/README.txt.

### Judge budget and board

Codex UTC-day 2026-10-23 usage: **0/8 checks, 0/3 scored runs**. No jobs submitted,
so no withdrawals. No H200 run was justified by the local native timings.

At ~20:48 UTC the shared account had used 6/20 checks and 3/10 scored runs,
leaving 14 checks / 7 scored. Our board best remains **1.9207x**, job 740bc5e9,
rank 31 at that read. The leader had moved to 2.7257x.

Read the result of Claude's previously pending check **1985ba311f6b480a**:
16/16 pass, **1.8960x**, CI **[1.8676, 1.9245]**. This was Claude's job, not a new
Codex check. It did not exceed the existing 1.9207x best, so it does not authorize
a scored submission under the same-code/check-beats-board rule. Left Claude's
pending entry unchanged in its owned notes/coordination lines; reported the result
in Codex's own handoff instead.

### Next cycle

1. Avoid another small tile/warp/stage sweep of this native design. It remains
   substantially slower despite removing much of the original instruction overhead.
2. If continuing full attention, prototype a gather-only native design that loads
   both KV heads as full contiguous 128-byte rows, then add attention while checking
   that the gather throughput survives. The present kernel handles one head per CTA;
   it never established a gather-only throughput ceiling for that topology.
3. Alternatively move to the other Codex direction: one-launch native sliding
   attention + final split reduction. Claude's latest notes show combine/prologue
   costs are the remaining opportunity there. Preserve the existing Triton plans.
4. Merge origin/main before new work, read ~/COORD.md, and claim only the native lane.
   /home/daytona/.okc/bin/okc is the CLI path (not on the non-login shell's PATH).
