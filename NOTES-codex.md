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

## Session 2 — 2026-10-23, 20:56–21:40 UTC (cycle complete; scored run queued)

### Candidate and branch state

Native CUDA output reduction for small sliding batches, built on main 0495b16.
Candidate commit **881df62**, now promoted to `codex`. Experiment source/results
are on `codex-exp/cuda-sliding` through **5169be7**; they are not imported by production.
The native function merges exactly 32 FP32 splits, one warp per query head.
Each lane computes a split's max/sum weight, the warp reduces the global max and
normalizer, and shuffles distribute weights to the two FP32 output dimensions
owned by each lane. RMSNorm/gate/output BF16 remain inside the timed call.
Dispatch is by shape only: sliding, S=32, B*Hkv <=4. All other shapes keep main's
latest Triton kernel and plans, including `_combine_hb` from 0495b16.

Five paired local H100 graph medians (cold L2, shared GPU flock):

| batch | q_len | main 0495b16 | native |
|---|---|---:|---:|
| 1 | 1 | 6.224 us | 5.664 us |
| 1 | 8 | 7.376 us | 6.928 us |
| 2 | 1 | 7.056 us | 6.496 us |
| 2 | 8 | 8.224 us | 7.760 us |

Final local full bench **1.838x** vs fixed SGLang/H200 reference. Entry baseline
before main's merge update was 1.812x; the gain includes both main's batched
Triton merge and our small-batch native path. Do not attribute the entire aggregate
gain to native code. Native timing benefit against the same-process latest-main
comparison is shown above. Files: `experiments/sliding/results/final-*`.

### Validation and a compiler-ordering bug caught locally

- Final production candidate: full reference suite 16/16 pass.
- Selected native B=1/2, L=1/8: four scored/nearby reference comparisons and 48 CUDA
  graph replays with changed QKV all passed.
- 80 additional graph replays changed every tensor (including cache contents,
  page tables, sequence lengths, positions, cache locations, scales), using
  mixed lengths, soft/edge distributions, pages 1/64, scales 1 and 1/64: all pass.
- Memcheck and synccheck: zero errors on four native cases, three changed-QKV
  calls each, pages 1/64, L=1/8, scales 1 and 1/64.
- Before narrowing dispatch, the corrected broader native merge passed 20 nearby
  sliding shapes (B=2–64, contexts 40K–120K) and four short-context DP cases. Larger
  verify shapes did not beat main's newer merge, so final dispatch retains main.

The first merge used const/restrict partial pointers. NVCC moved read-only
`LDG.E.64.CONSTANT` loads before `griddepcontrol.wait` (`ACQBULK` in SASS), despite
an asm memory clobber. Replaying identical inputs concealed the stale reads;
nearby B=48 and repeated eager calls exposed NaNs/large errors with PDL enabled.
Turning PDL off avoided the failure. Volatile inline `ld.global.cg.v2.f32` after
the dependency wait fixes the ordering; final SASS confirms wait-before-partials.
The fresh-value and fresh-routing graph tests above validate the fix.

**Discard early native merge timings** (~4.9 us decode-1, 1.879x aggregate).
They came from the incorrect version. That code never reached a judge check or
scored run. All judge work uses the corrected cooperative merge. Nothing needed
withdrawing. The archived invalid results are explicitly labeled in the README.

### Native attention experiments that did not win

1. Warp-local CUDA sliding decode: cp.async gather ring, FP16 QK/PV MMA,
   fused norm/RoPE/new-FP8-cache writes, FP32 split partials. Swept S=2/4/8/16,
   W=1/2/4, stages=2/3, fused last-CTA or separate Triton merge.
   Best valid two-launch times ~11.1/15.9/20.8 us for B=1/16/32 (slower than main).
   Fused last-CTA reduction lost. All swept scored cases passed comparisons.
2. Hopper cluster/DSM reduction of those warp-local partials: correct on scored
   sweeps but best ~13.4/16.8/23.5 us. Eliminating global partial traffic alone
   does not fix the slow attention loop.
3. WGMMA sliding verify adapted from session 1, with cluster reduction: initially
   failed with intermittent NaNs. Separate shared partial storage, atomic flags,
   and synchronous cooperative gathers did not resolve it. The actual issue was
   descriptor encoding: the matrix address must be `(address & 0x3ffff) >> 4`.
   Omitting the mask can let cluster address bits corrupt the descriptor. The fix
   passed all 24 scored shape/plan comparisons, eight cooperative diagnostics, and
   six additional small/DP/page/scale cases at both splits 4 and 8.
   Best correct times ~24.4/30.4/39.6 us at B=1/16/32, still much slower than main.
   An early racecheck on the unmasked pipeline reported hazards; no clean
   racecheck has been claimed for this prototype. Never dispatched or submitted.
4. Native merge with compile-time head/length constants: no material improvement.
   Cooperative per-warp weights beat repeated per-lane exponentials for small
   verify batches; neither merge design reliably beat main on large verify grids.

Native decode profile (B32, S8, W4, stages2): 96 regs, 37.3 KB shared memory/CTA,
5 theoretical CTAs/SM but only 0.78 grid waves, 21.6% achieved occupancy,
26% DRAM utilization, ~0.30 issued warps/scheduler. No spills. Latency/occupancy,
not a raw HBM ceiling, is the issue. Full text profile and reproducer retained.

### Judge budget and result

Codex UTC-day 2026-10-23: **1/8 checks**, **1/3 scored runs**.
Check **2d178f1df3544c27**, code 881df62, submitted 21:17 UTC: **16/16 pass,
1.942450x**, 95% CI **[1.913313, 1.971587]**, no flags or static-check rejections.
The check completed at about 21:37 UTC after ~13 minutes queued and 6 minutes running.
Scored run **033117b46b6f42ba** submitted 21:37 UTC, identical checked code: **pending**.
Immediately before submission the board best was 1.9207x, rank 32, so the required
same-code/check-beats-current-board condition was satisfied. This is Codex's first
scored run today; the human's/Claude's earlier runs are separate agent shares.
Before it, account usage was 7/20 checks and 3/10 scored; the eighth account check
is ours. Claude's main-merge check 944b20a9 was running concurrently. At session
end the shared account had used **8/20 checks and 5/10 scored runs**, leaving
12 checks and 5 scored runs; Codex's own usage remains 1 check and 1 scored run.

Read `profiles/submission.2d178f1df3544c27.txt`. The native-path shape times
were decode-1 sliding **7.185 us** and verify-1 sliding **7.951 us**. Main 0495b16's
check 944b20a9 measured **7.482 / 8.430 us** on those shapes and **1.934634x**
[1.905615,1.969061] overall. These are separate judge runs, so treat the small
aggregate difference cautiously; the confidence intervals overlap. The paired
H100 comparisons and both H200 target shapes favor the native path.

The native merge itself was profiled at **1.6 / 1.9 us** for decode/verify batch 1;
the attention kernels cost 4.6 / 4.3 us. Large full-verify attention remains the
largest weakness (156.7 / 306.1 us for B16/32 in this check). PDL profile rows for
full shapes overlap and include waiting; do not add them to estimate total time.

Neither ~/STOP nor ~/STOP-codex existed at the last read. No withdrawals: the
incorrect early prototypes were never checked or submitted. Read the scored
run's result and update this entry when it completes. The scored run was still
queued at the final 21:40 UTC read. Local work, H200 validation, promotion, and
submission are complete; the cycle ends without waiting idle through another queue.
A final fetch confirmed origin/main was still 0495b16 and already merged. Both
codex and codex-exp/cuda-sliding are pushed, with a clean production worktree.

### Next cycle

- First read `okc status 033117b46b6f42ba --json` (or `--wait --json` once running),
  read its profile if it passed, and update the job result/budget/board in these
  notes and Codex's own COORD.md line. Do not submit the same code again.
- Start from the final production branch and merge any newer main before tuning.
- Do not repeat tiny warp/stage sweeps of the current full native attention
  prototypes: they are substantially slower. Their source/measurements are on
  codex-exp/cuda-sliding under experiments/sliding, with explicit correctness labels.
- Native merge gains are small. Future larger wins need a different gather/
  attention topology; the pipeline/full-row-load idea from session 1 remains
  structurally more promising than another small merge parameter sweep.
- For any native kernel synchronized with a concurrent predecessor, inspect SASS
  for data loads before waits, and validate fresh data/routing across graph replays.

## Session 3 — 2026-10-23, 21:40–22:07 UTC

### Outcome

No production change from the new native full-attention experiments. A measured
full-row gather ceiling is now established, and a correct Hopper mbarrier pipeline
is retained, but attention still loses substantially to main. No new judge checks
or scored runs were used. The prior session's scored run completed successfully:
**033117b46b6f42ba, code 881df62: 16/16 pass, 1.945619x,
95% CI [1.916435, 1.974803]**. No flags or static-check rejections. It is the account's
new best, **rank 31**, at the 22:04 UTC leaderboard read; to_first=1.401.

The experiment is `codex-exp/cuda-gather`, source/results commit **40b56db**.
Start with `experiments/gather/README.txt`; `pair_mbar.py` is the strongest retained
native prototype and `compare_final.py` reproduces its final comparison. All
experiments remain outside production imports. `origin/main ac18ec3` was merged;
its changes since 0495b16 are notes/profiles only, so the production kernel is
byte-for-byte the same checked/scored 881df62 kernel. Codex's small sliding merge
remains dispatched exactly as in session 2.

### Gather-first measurements and native attention results

All times below are local H100 cold-L2 CUDA graph replay times with the shared GPU
lock and empty-replay subtraction. They are not judge H200 measurements.

- A pure native checksum gather of both 128-byte K/V head rows reaches **128 / 246
  us** at batches 16/32 (cooperative four-warp copies, 3 stages, ~1056 CTAs). This
  reproduces the old ~236 us uniform-context gather floor reasonably closely on
  the actual agentic scored inputs. A dedicated single producer warp gives about
  **156 / 308 us** at that grid. Two producers can reach ~258 us at B32 with more
  CTAs; four producers lose residency and bottom out around 299 us.
- Two consumer warpgroups (one per KV head), with full-row gathers from 1/2/4
  producer warps, give best B32 attention times **563 / 497 / 544 us**. Cooperative
  8-warp loading is ~595 us. None transfers the gather-only advantage to attention.
- Replacing software-ready flags and wait_group publication with Hopper
  mbarriers improves the two-producer attention version to ~438 us in the initial
  sweep. The retained version has 64 producer arrivals tracked by
  cp.async.mbarrier.arrive.noinc and one empty-stage arrival from each consumer
  group. New-token shared stores use release ordering and the async-proxy fence.
- A single consumer group running both softmax chains still costs ~497 us.
  WGMMA PV, cooperative loads, and page-index broadcasts do not rescue it;
  hardware barriers on this dual-chain variant are ~495 us. Do not repeat this
  family of small parameter sweeps without a new resource/issue-throughput idea.
- Linear bulk asynchronous row-copy gather alone reaches ~260 us. Direct tensor-map
  row gathers into a SW128 WGMMA K layout are correct, but attention is ~841 us at
  B32. Linear bulk K/V plus explicit shared K rearrangement is ~872 us. These copy
  engines need much more resident gather work than the attention kernel permits.

Final alternating-order comparison, medians of three paired measurements:

| batch, L=8 full | production | native mbarrier |
|---|---:|---:|
| 1 | 18.720 us | 34.864 us |
| 8 | 88.080 us | 127.216 us |
| 16 | 158.064 us | 231.296 us |
| 24 | 226.944 us | 330.896 us |
| 32 | 307.264 us | 447.072 us |

The native version remains about 46% slower at the target large batches. Keep
production. Its full local bench was **1.827x** against the fixed SGLang/H200 times,
with verify 16/32 full **156.26 / 306.21 us**. This is timing variation in unchanged
code, not a speedup or regression introduced in this cycle.

NCU on the four-producer paired prototype: 80 registers/thread, ~52 KB shared
memory, two resident CTAs/SM, 35.7% achieved occupancy, 32.4% DRAM throughput,
0.53 issued warps/scheduler. Long-scoreboard, short-scoreboard and execution wait
cost about 2.1 cycles each per issued instruction; barriers ~0.86. Full-row loads
alone do not overcome the resource, dependency and instruction overhead. Profile
text, CSV counters and the lock-taking reproducer are on the experiment branch.

### Validation

- Software-flag native/fallback dispatch passed 16/16; 35 extra fresh-data graph
  replays passed on seven nearby shapes (B1/2/3/4/8/12, TP8/DP, pages 1/64,
  contexts 10–120000, scales 1 and 1/64, soft/coherent/edge inputs).
- Retained hardware-barrier dispatch separately passed 16/16 and those 35 fresh
  graph replays. Another 35 replays also varied sequence lengths, positions,
  cache locations and cache_seqlens on device, including empty contexts: all pass.
- Hardware-barrier **memcheck and synccheck: 0 errors; racecheck: 0 hazards,
  0 errors, 0 warnings**. Four small page/scale cases, three native calls each.
- Direct and bulk checksum gathers independently matched PyTorch byte checksums
  on 128 combinations each, including DP heads and page size 64. An early log
  incorrectly printed 64 variants per shape; it ran 32 per shape (four shapes).
  The script/README and later bulk log state the correct count.
- Every attention sweep checked finite outputs and <2.5% difference against a
  reference-validated production output (observed ~0.03% or less). Other
  prototypes have only these scored comparisons; do not claim full-suite or
  sanitizer validation for them.
- Production separately passed 16/16. Main's later merge only changed prose and
  profiles, so no redundant GPU test rerun was needed for that merge.

### Prior scored run, profile and budget

The run queued at 21:37 began at **21:57:36 UTC**, completed around 22:04, and was
read during this cycle. It was not resubmitted. Its profile is saved at
`profiles/submission.033117b46b6f42ba.txt` (with JSON/log alongside).

H200 native-path sliding shape times: **decode 1 = 6.93 us; verify 1 = 8.26 us**.
The native merge's profiled kernel durations are **1.7 / 2.0 us**, attention
**3.5 / 4.2 us**. Full verify 16/32 remain the weak spots at **155.26 / 304.20 us**,
1.316x / 1.277x. PDL profile rows overlap and include predecessor waiting; do not
sum them. The aggregate 1.9456x is consistent with the same-code 1.94245x check.
It is only ~1% above main's 1.927x board run, inside the confidence intervals;
the earlier same-process native-vs-main measurements are stronger evidence for
the small merge improvement than this difference between independent judge runs.

Codex UTC-day 2026-10-23 usage remains **1/8 checks, 1/3 scored runs** (both session
2). This cycle: **0 checks, 0 scored**. Account at 22:04: **8/20 checks, 5/10 scored**,
leaving 12 checks / 5 scored. No withdrawals; no experimental code was submitted.

### Next cycle

1. Start from `codex`, fetch/merge main, and read the shared board. The pending
   session-2 job is now resolved; do not poll or resubmit it again.
2. Do not repeat the native full-row gather/producers/TMA sweeps: the gather
   ceiling is good, but every attention topology tried here loses. The mbarrier
   prototype is a useful correct building block, not a candidate.
3. A larger native attention attempt needs to demonstrate reduced per-tile
   instruction/dependency cost at the required register/shared-memory budget.
   More gathered bytes in flight alone has not solved it. Claude currently owns
   Triton per-tile instruction work and V FP8-to-FP16 staging; respect that claim.
4. A smaller distinct direction is a native hierarchical reduction for the
   batch-1 full layers' ~128 splits (their merge costs ~2–3 us). The current native
   path handles only 32 splits on small sliding batches. Preserve FP32 partials,
   correct PDL load ordering, and judge existing shape plans without changing them.
5. Retain fresh routing/length graph tests and sanitizer checks whenever changing
   asynchronous synchronization. For experiments use the GPU lock for profiling
   and correctness stress runs as well as every timed replay.


## Session 4 — 2026-10-23, 22:07–22:30 UTC (local cycle complete; checks queued)

### Outcome and source state

A new native hierarchical reduction wins on small full-attention batches.
Production source matches **effc2bf** on `codex-exp/cuda-reduce`: full layers,
B*Hkv<=4, and 64<actual splits<=128. Eight warps each merge up to 16 FP32 splits,
then warp zero combines their local maxima, denominators and vectors from
shared memory. RMSNorm, gate, BF16 output and every input-derived calculation
remain inside kernel_fn. Native attention kernels are still not dispatched.

Merged main **651b421** (one-pass sliding merge) and **65ff1ab** (new tokens
before cached keys). All Triton changes/plans belong to main. The new sliding
merge beats our session-2 native sliding path, so that old native code and
binding were removed. Paired H100 improvements for main over the old native
sliding path were B1 decode 5.568 -> 5.312 us, B1 verify 6.816 -> 6.368 us,
B2 decode 6.176 -> 5.792 us, B2 verify 7.424 -> 7.088 us. Retain main here.

The experiment source and results are on `codex-exp/cuda-reduce`, starting with
`experiments/reduce/README.txt`. The source before the later main merge was
**fddd5c1**; its result archive starts at **5930643**. Experiments are not imported
by production. The final candidate is locally validated; its H200 confirmation
is still pending. No scored run was requested during this cycle.

### Local measurements

All times are H100 CUDA graph replay, cold L2, empty-replay subtraction, shared
GPU lock. The same-process comparisons alternate order and take medians of
five paired measurements. The latest native-vs-main full-layer comparison:

| batch | q_len | context | main 65ff1ab merge | native |
|---|---|---|---:|---:|
| 1 | 1 | 80000 | 17.904 us | 16.928 us |
| 1 | 8 | 80000 | 19.136 us | 18.400 us |
| 2 | 8 | 80000 agentic | 40.032 us | 38.912 us |
| 2 | 1 | 1500, page 64 | 7.264 us | 6.656 us |

The full local bench is **1.909x** versus fixed SGLang/H200 times. Earlier in
the cycle, with main 651b421, it was 1.882x; before that main merge, 1.835x.
These aggregate changes include main's improvements. Do not attribute them
entirely to native code or compare them as if measured simultaneously. The
paired table isolates the native reduction's contribution. Earlier paired
small full-shape gains were generally ~0.3–0.8 us decode, ~0.7–0.9 us verify.

The wider 14-case scan covered B1/B2/B3, contexts 4K–120K, pages 1/64 and both
query lengths. No new slowdown appeared on nearby shapes. Short B2 decode
at contexts 1024/1500 also uses this native path and passed independent reference
checks; context 2000 agentic exceeds its split limit and retains Triton.
The B2/B3 verify timing discontinuity already exists in main's attention plans;
this change reduces B2 time and leaves B3 unchanged.

### Experiments and lessons

- A warp per head handling 64/128 splits needs many registers and loses several
  microseconds. Four/eight cooperating warps reduce this pressure and win on
  small grids. Native gains at large batches were small and inconsistent;
  final dispatch keeps the inherited full-layer merge there.
- Split/dimension tiles (8/16 lanes per row), with vectorized 16-byte partial
  loads, lost to the simpler two-float-per-lane hierarchy.
- CUDA cluster/DSM reduction spreads one head over SMs but loses to an ordinary
  block. It has scored-shape comparisons only, not full-suite/sanitizer coverage.
- Compile-time head/token indexing removes a substantial integer-divide prologue
  but does not improve total time. PDL already overlaps that prologue with the
  predecessor. Do not spend another cycle on this specialization alone.
- A late cooperative-weight version of warp zero tied verify and appeared ~0.4 us
  faster on decode in one small comparison. It is **not integrated or submitted**,
  and is not established as a win. It only has six baseline comparisons, three
  with independent reference checks. It could be reconsidered next cycle if a
  small follow-up is worthwhile; do not claim full validation for it.
- NCU standalone B1/L8/S126 reduction: native 48 registers/thread, 2176 shared
  bytes, 11.79% achieved occupancy; Triton 80 registers/thread, 1024 dynamic
  shared bytes, 6.22%. Cold standalone times 5.50/5.57 us are not production
  PDL graph timings: partials are cold in this isolated profile. Use the paired
  kernel_fn measurements for the end-to-end benefit.

### Correctness and ordering

- Generic hierarchy: 16/16 reference tests, 72 fresh-data/routing/length graph
  replays, and 162 independent synthetic-partial comparisons with changing
  weights, vectors, gates and scales.
- Integrated fddd5c1: 16/16 and 72 fresh graph replays. After main 65ff1ab merge,
  effc2bf separately passed 16/16 and another 72 fresh graph replays.
- Final native merge isolated under Compute Sanitizer: **memcheck zero errors;
  synccheck zero errors; racecheck zero hazards/errors/warnings**. Each run checks
  42 fresh graph replays across seven shapes against torch, including DP heads,
  unaligned split counts and 1/64 scales (some exercise the implementation beyond
  its narrower dispatch).
- Final PDL SASS shows ACQBULK before every coherent partial load. Volatile inline
  PTX ld.global.cg.v2.f32 keeps nvcc from hoisting those reads. Gate and scale
  loads before the wait are independent of the predecessor and safe.
- The first all-kernel memcheck log reported two CUDA_ERROR_INVALID_VALUE calls
  from Triton's loader, not memory accesses. Local driver.c lines 313–328 probes
  the newer oversized-shared-memory attribute and intentionally falls back on
  failure. Keep that log accurately labeled; it is not a zero-error log. The
  isolated production-native sanitizer runs above avoid that unrelated loader
  probe and retain full memory/sync/race checking.
- The initial sweep tried an unsupported 256-split one-warp configuration and
  caught a mismatch. Later experimental launchers explicitly check capacity and
  the sweep skips unsupported configurations. Production handles at most 128
  splits, with exactly 128 slots of capacity. No bad prototype was submitted.

### Judge jobs and budget

Both checks were still **queued at 22:28 UTC**:

| job | requested | source | status |
|---|---|---|---|
| 40e1aa41415a4c87 | 22:19 UTC | fddd5c1, main 651b421 | queued |
| 01bc4ecc9c7047d9 | 22:27 UTC | effc2bf, main 65ff1ab | queued |

The second check is necessary because the later main merge changed the source.
Only **01bc4ecc9c7047d9** can qualify the latest source for a scored run. Read the
older result for evidence too, but do not score the new source using its result.
Do not resubmit either check. No new scored runs, no withdrawals.

Codex UTC-day 2026-10-23 usage: **3/8 checks, 1/3 scored runs**. This cycle used
2 checks and 0 scored. Account was 11/20 checks, 5/10 scored immediately before
the second check; it therefore had 8 checks and 5 scored left afterward, absent
other concurrent jobs. Refresh limits before any further judge action.
The last read board best was the prior **1.945619x** run 033117b46b6f42ba; do not
assume it remains the best when the pending checks finish.

### Next cycle

1. Read both pending results and each passing profile. Latest source is effc2bf;
   compare its check to the account's **current** best before any scored run.
   A scored run is allowed only if this exact code's check beats that best.
2. If the latest H200 check supports the local win, submit the identical source
   (provided budget and current board permit). If H200 contradicts it, retain
   main 65ff1ab improvements and remove/rework only the native full dispatch.
3. Read COORD.md and fetch main. A further main code change means a further
   combined-code check is needed before scoring; do not confuse source versions.
4. Native hierarchical merge is a small improvement, not the large full-verify
   breakthrough. Avoid repeating the earlier native attention/gather topologies
   without a concrete reduction in per-tile instruction/dependency cost.
5. If exploring cooperative.py, validate fresh PDL replays and repeat paired
   timing before considering a candidate; its one decode result is inconclusive.

No STOP file existed at the final local-work read. The local cycle ends with
clear queued-job handoff rather than sleeping idle through the judge queue.


Final handoff at **22:30 UTC**: promoted the locally validated source to `codex`
commit **5e2441f** and pushed it; its submission.py git blob is exactly identical
to effc2bf (the source checked by 01bc4ecc9c7047d9). Experiment/results branch
`codex-exp/cuda-reduce` is pushed through **0c458e9**. Both checks are still queued.
Account usage is confirmed **12/20 checks, 5/10 scored**, leaving 8/5; Codex's
own usage remains 3 checks/1 scored. Current account best is still 1.9456x,
now rank 32. Neither STOP file exists. No scored run was authorized by a new
check result this cycle, so none was attempted. Production is locally validated,
with H200 confirmation pending; next cycle must resolve that before scoring.

## Session 5 — 2026-10-23, 22:31–22:41 UTC

### Outcome

No new production change or judge jobs. The small cooperative-weight follow-up
from session 4 did not reproduce its apparent win. Six native reduction variants
were compared, and all are retained as experiments only. The existing native
hierarchy still wins against main in a tighter same-process paired comparison.
This was a short focused follow-up: useful local work finished without a candidate,
so the cycle does not wait idle for the judge queue.

Experiment branch **codex-exp/cuda-reduce-coop**, source/results **0fb4c69**.
Start at `experiments/reduce/session5/README.txt`. Session-4 experiment history
was merged into that branch; the notes conflict retained this branch's newer
session-4 handoff. Production submission.py is unchanged from **5e2441f**, exactly
the source of **effc2bf** checked by 01bc4ecc9c7047d9. Its file SHA256 is
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`
(the judge content hash is a different identifier). Main remained **65ff1ab**
at the last fetch and is already merged. Do not merge the experimental files
into production imports.

### Measurements and lessons

All measurements are H100 cold-L2 CUDA graph replays with the shared GPU lock,
empty-replay subtraction, and unchanged Triton attention/plans. The new paired
runner captures each graph once, rotates/reverses execution order, and records
100 rounds per case. Each variant also passes the independent reference on the
six measured scored/nearby cases. Median paired deltas need not equal the
difference between two independently computed medians.

- **Cooperative final weights:** the old five-pair method gives production ->
  candidate **17.216 -> 17.296 us** decode B1, **18.304 -> 18.432** verify B1,
  **38.816 -> 38.992** verify B2. Tighter paired deltas are **+0.144 / +0.240 /
  +0.208 us**. The earlier isolated ~0.4 us decode gain does not reproduce.
- **64-bit address/stride calculation; unmasked complete warps:** generally
  lose ~0.25–0.46 us on the small scored grids. One B2 verify win is insufficient
  for a new dispatch. No precision or attention arithmetic changes.
- **Gate sigmoid before PDL wait:** an empty volatile asm consuming the results
  does not constrain ptxas scheduling enough. SASS still puts all eleven EX2
  instructions after ACQBULK, none before it. This is not actual early arithmetic.
- **Gate sigmoid through volatile shared storage:** genuinely moves two EX2 and
  two STS before ACQBULK; all coherent partial loads remain after the wait. Paired
  B1 decode/verify changes are only **-0.096 / +0.048 us**. Nearby results vary.
  No convincing scored-load improvement; not promoted or submitted.
- Existing production versus inherited main remains a clear local win:

| full shape | main merge | existing native | paired native saving |
|---|---:|---:|---:|
| B1, L1, 80K, page1 | 16.928 us | 16.272 us | 0.656 us |
| B1, L8, 80K, page1 | 18.464 us | 17.808 us | 0.672 us |
| B2, L8, 80K agentic, page1 | 39.184 us | 38.496 us | 0.672 us |
| B1, L1, 40K, page64 | 11.264 us | 10.704 us | 0.544 us |
| B1, L8, 120K, page64 | 24.336 us | 23.840 us | 0.640 us |
| B2, L1, 1500 agentic, page64 | 6.576 us | 6.224 us | 0.368 us |

Do not infer an aggregate speedup from runs taken at different times. No new
full production bench is claimed: production code is unchanged and the native
contribution was isolated in the paired comparison above.

### Validation

The cooperative candidate passed **16/16** independent reference tests and
**72 fresh-data/routing/length graph replays** across nine native/fallback cases,
including shortened/empty contexts, pages 1/64 and scales 1 and 1/64. This does
not mean 72 native-only calls. Other variants have six independent reference
comparisons each and repeated graph timings, not full-suite/fresh-data/sanitizer
coverage. No variants were submitted. The unchanged production code retains
session 4's full validation and sanitizer results.

The first new paired-driver attempt examined capture outputs before replaying
the graphs, producing NaNs from uninitialized buffers, including the unchanged
production baseline. The driver now replays every graph before checking its
outputs. The incomplete log is labeled `paired-harness-incomplete.log`; no
measurements from that attempt are used. It was a driver error, not a kernel
correctness failure or a submitted implementation.

### Judge status, budget and next cycle

At **22:40 UTC**, older check **40e1aa41415a4c87** (fddd5c1) was **running**;
current-source check **01bc4ecc9c7047d9** (effc2bf) was still **queued**. Both were
read repeatedly, neither was resubmitted, and neither yet had a passing profile.
Only 01bc4ecc9c7047d9 can qualify the current production source for scoring.

Codex UTC-day 2026-10-23 usage remains **3/8 checks, 1/3 scored**. This cycle:
**0 checks, 0 scored, no withdrawals**. Shared account at 22:40: **12/20 checks,
5/10 scored**, leaving 8/5. Best scored result remains **1.945619x**, rank **32**,
job 033117b46b6f42ba. Refresh the board and limits before taking any judge action.

1. Read both pending jobs and passing profiles. If the latest exact-code check
   beats the then-current account best, submit that identical production source
   within the remaining budget. Otherwise retain/reassess its native dispatch
   using H200 shape evidence; do not score using the older source's check.
2. Fetch/merge main before any new code. A later source change requires its own
   combined-code check before scoring.
3. Close the cooperative-weight follow-up: it is correct but not faster. Avoid
   further small native merge sweeps absent a new structural mechanism; the
   remaining tenths of a microsecond are inconsistent and low payoff.
4. Larger native attention work still needs a concrete reduction in per-tile
   instruction/dependency cost. The previous gather/producers/TMA designs and
   small merge variants have now been measured extensively; do not repeat them.

Neither STOP file existed at the last read. Experiment results and production
notes are pushed separately; the production kernel remains locally validated
with its H200 confirmation pending.

Final 22:41 UTC read: 40e1aa41415a4c87 still running, 01bc4ecc9c7047d9 queued.
Status snapshots are saved in profiles/session5-{older,current}-status.json.
Experiment branch is pushed through b8246f7; production notes c9d4535 and
this final snapshot commit are pushed. No source change, new job, or STOP file.

## Session 6 — 2026-10-23, 22:42–22:54 UTC

### Outcome and source state

No production code change and no new judge jobs. Two concrete native ideas lost
locally: separate whole-cache materialization, and direct FP8 V operands without
shared FP16 staging. Both are archived on **codex-exp/cuda-pack**, source/results
commit **ed5d2c9** plus a generated-log whitespace cleanup. Start with
`experiments/pack/README.txt` and `experiments/direct_v/README.txt` on that branch.
The experiments are not imported by production. This focused cycle ended after
decisive measurements and validation, rather than waiting idle for the judge queue.

`origin/main` remained **65ff1ab**, already merged. Production submission.py is
unchanged from **5e2441f/effc2bf**, SHA256
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No native attention prototype is dispatched. Existing small-full native reduction
and main's Triton attention/sliding merge remain in use.

### Pending-check progress and profile

Older check **40e1aa41415a4c87**, source **fddd5c1** (main 651b421 plus native full
reduction), completed and was read at 22:44 UTC:

- **16/16 passed**, **1.984296x**, CI **[1.954532, 2.022082]**, no flags/rejections.
- H200 small-full times: decode 1 **15.400 us**, verify 1 **16.459 us**; profiled
  native reduction **1.8 / 2.4 us**. Full verify 16/32 **154.072 / 302.687 us**.
- Read and saved `profiles/submission.40e1aa41415a4c87.txt`, with JSON/log alongside.
  PDL rows overlap and include predecessor waiting; do not sum those kernel times.
- This exceeds the current account best, but **cannot qualify the newer current
  source for scoring**. Only current-source check **01bc4ecc9c7047d9** can do that.
  It was still **queued at 22:52 UTC**, with no result or passing profile.

No check was resubmitted and no scored run was attempted. The old check supports
the native small-full reduction but does not isolate its contribution from main's
sliding merge. Session 5's paired native/main comparison remains better evidence
for that component than independent aggregate judge results.

### Native whole-cache packing: reject before dense attention

A native kernel gathers every historical K/V row into contiguous FP8 scratch,
bit-for-bit. It changes no precision and copies fresh cache/routing on every call.
This would be a necessary prefix of gather-then-dense attention; it does not yet
compute attention or insert new tokens. At large batches, that prefix alone loses:

| full verify batch | best copy-only | complete production call |
|---|---:|---:|
| 1 | 16.288 us | 18.272 us |
| 16 | 246.816 us | 154.848 us |
| 32 | 494.144 us | 305.488 us |

All are local H100 cold-L2 CUDA graph medians, empty replay subtracted, shared GPU
timing lock. Copy numbers are minima from 36 configurations per case, not paired
speedup estimates. Vector widths 4/8/16 bytes, unroll 1/2/4/8, grids ~264/528/1056
CTAs were measured. Source byte counts were 20.48 / 330.70 / 662.06 MB; the output
adds another full-cache write. Installing/implementing dense FA3 was not justified
once the copy alone cost ~60% more than the complete production large-batch call.

Validation: all **108** swept configurations passed bit-exact cache comparisons;
**288** fresh graph checks on eight small/nearby shapes passed, including TP8/DP,
pages 1/64, changed cache bytes/page routes, zero lengths and unchanged output
suffix sentinels. These are copy checks, not attention tests. No packing sanitizer
coverage is claimed, and this experimental copy uses 32-bit offsets for the tested
allocation sizes. No production dependency was added.

### Direct FP8 V registers: correct, but more instructions

Modified the session-3 paired-head/mbarrier prototype only. Load FP8 byte pairs
with ldmatrix.trans, shuffle the required column pairs to each lane, permute their
bytes, convert exactly to FP16 register operands and run the same PV MMA. This
removes the shared FP16 V array, cooperative conversion and a consumer barrier.
All keys, cache writes, query rounding, FP32 partials and softmax are preserved.
No Triton source or plans were tuned.

Same-process cold-L2 graph timings at four stages (sequential comparisons, not a
balanced paired estimator):

| full verify batch | prior native | direct V | production |
|---|---:|---:|---:|
| 1 | 34.576 us | 35.152 us | 18.400 us |
| 16 | 231.152 us | 239.872 us | 155.072 us |
| 32 | 446.672 us | 454.080 us | 301.552 us |

Stages 3/5 do not rescue it. After the full reference suite, the complete
native/fallback bench was **1.585x** against fixed SGLang/H200 reference times.
That is an experimental local benchmark, not a production regression or judge run.

S4/page1 SASS explains the failure: removing staging saves 8 LDS.U16, 8 STS and
8 of 16 LDSM, but adds **32 SHFL.IDX + 32 PRMT** and raises FP8-to-FP16 conversions
from **8 to 32** per tile. The shared-memory saving (62,528 -> 53,312 bytes) does
not improve residency: registers rise **94 -> 96**, still two CTAs/SM. Static
SASS counts and the two relevant function dumps are archived; no dynamic Nsight
instruction-counter measurement is claimed. The byte-pair fragment mapping was
checked against NVIDIA's PTX mma.m16n8k16 documentation (linked in the README).

Validation:

- 15 independent reference comparisons: five shapes at stages 3/4/5.
- Full reference suite **16/16**; five native cases, eleven fallback cases.
- **42 all-native fresh graph replays** across seven nearby shapes, including
  pages 1/64, TP8/DP, scales 1 and 1/64, soft/coherent/edge data, contexts 10–120K.
  Fresh cache/QKV/routing, changed lengths/positions/cache locations, and empty
  contexts all passed independent reference checks.
- Native attention isolated under Compute Sanitizer: **memcheck 0 errors,
  synccheck 0 errors, racecheck 0 hazards/errors/warnings**. Four small page/scale
  shapes, three changed-QKV calls each. The sanitizer driver checks finite
  partials; full output correctness is covered by the reference runs above.
- All run processes completed successfully. No experimental source was submitted.

### Budget, next cycle, and stopping state

Codex UTC-day usage remains **3/8 checks, 1/3 scored**. Session 6 used **0 checks,
0 scored**, no withdrawals. Account at 22:52: **12/20 checks, 5/10 scored**, leaving
8/5. Current account best **1.945619x**, rank **32**, job **033117b46b6f42ba**.
Status/board/budget snapshots are in `profiles/session6-*.json`.

1. Resolve **01bc4ecc9c7047d9**, without resubmitting it. Read its passing profile;
   score the exact checked current source only if its result beats the then-current
   account best and budget permits. Existing production already passed local
   16/16 plus extensive fresh-graph/sanitizer validation in sessions 4/5.
2. Fetch/merge main before new code. A later source change requires another
   exact-source check before scoring; old **40e1aa41415a4c87** is not sufficient.
3. Close both materialization and direct-register-V directions: the former pays
   an entire extra cache write, the latter multiplies conversions and shuffles.
   Avoid repeating the previous native pipelines or tiny merge sweeps without a
   concrete change that reduces per-tile work or increases useful residency.
4. Keep experiments on their branch. Production source is unchanged and locally
   validated; its exact combined-code H200 confirmation is still pending.

Neither STOP file existed at the final work read. Notes, older passing profile,
and experiment results are pushed separately; the cycle ends on `codex`.

Final read **22:54 UTC**: current-source check **01bc4ecc9c7047d9** remains queued;
no result to score. Production SHA256 reconfirmed unchanged, no STOP files, clean
worktree before this handoff append. Experiment branch pushed through **3678a78**,
production notes/profiles through **3e24ba0**. Shared claim removed and results
posted to COORD.md. No new judge job or budget use.

## Session 7 — 2026-10-23, 22:55–23:05 UTC (local cycle complete)

### Outcome and source state

No new production code. Producer-side FP8 V conversion was implemented in two
forms and rejected by cold CUDA graph timings. Sources, scripts and all measured
results are pushed on **codex-exp/cuda-producer-convert**, commit **f2e241e**;
start with `experiments/producer_convert/README.txt` on that branch. No experimental
module is imported by submission.py. This focused cycle ends after the mechanisms
were measured and validated rather than waiting idle for the judge.

`origin/main` remains **65ff1ab**, already merged into `codex`. Production source
is still the exact **5e2441f/effc2bf** code checked by **01bc4ecc9c7047d9**, SHA256
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No native attention prototype is dispatched. Main's attention/sliding merge and
the existing small-full native reduction remain unchanged.

### Native conversion results

Both designs preserve FP8 cache bytes, FP8 query rounding, FP16 value operands,
FP32 partials, cache writes, all-key attention and fresh-input graph behavior.
They only change where FP8 V is converted and how that work is synchronized.

1. **Direct producer conversion:** asynchronous K gathers plus aligned uint4
   V loads in the producer warps, packed FP8-to-FP16 conversion, and staged shared
   FP16 V. Separate completion barriers publish async K and generic V stores.
   Removes the conversion loop and its barrier from the attention consumers.
2. **Dedicated conversion warps:** preserve async K+V gathers, add two or four
   conversion warps between gather producers and attention consumers. Converted
   V has its own shared-memory ring and completion barrier. No consumer staging.

Same-process H100 cold-L2 CUDA graph medians, stage4, empty-replay subtraction,
shared GPU timing lock. The tables are separate sequential comparisons, not a
balanced paired estimator; gaps are large enough to reject the designs.

| direct conversion run | verify B1 | verify B16 | verify B32 |
|---|---:|---:|---:|
| production | 18.480 us | 156.544 us | 306.368 us |
| original paired native | 34.736 | 230.672 | 442.080 |
| two direct producers | 47.104 | 396.368 | 773.024 |
| four direct producers | 36.192 | 241.488 | 467.808 |

| async conversion run | verify B1 | verify B16 | verify B32 |
|---|---:|---:|---:|
| production | 18.400 us | 154.704 us | 304.432 us |
| original paired native | 34.736 | 230.640 | 442.208 |
| two conversion warps | 39.648 | 269.856 | 522.752 |
| four conversion warps | 45.648 | 363.408 | 719.824 |

Direct producers swept 2/4 warps and 2/3/4/5 stages. Dedicated conversion swept
2/3/4 stages with two warps, then 3/4 with four warps. None approaches production.
Full native/fallback benches after reference validation were **1.592x / 1.515x**
for retained direct4 / async2 respectively, against fixed SGLang/H200 reference
numbers. These are rejected local experiments, not production regressions.

Resource changes at stage4/page1:

- Original native: 94 registers/thread, 62,528 shared bytes, 320 threads.
- Direct4: 80 registers, 69,728 shared bytes, 384 threads, no spills.
- Async2: 80 registers, 90,208 shared bytes, 384 threads, no spills.
- Async4: 72-register cap for 448 threads introduces spills and is much slower.

Final isolated NCU profiles: original/direct4/async2 achieve **29.00 / 35.76 /
33.88%** occupancy, but issue rates stay **0.56 / 0.55 / 0.56** instructions per
scheduler cycle. Barrier stalls fall **9.16 -> 5.50/6.12%**, short-scoreboard stalls
**16.19 -> 8.64/9.75%**, while long-scoreboard stalls rise **13.93 -> 31.87/31.10%**.
These whole-kernel counters do not identify the exact stalled load. The changes
reduce consumer work without sufficiently reducing total work/dependencies.
Do not use NCU durations as graph timings.

The first profile collection could overlap unlocked benchmark warmups/captures,
even though all timed loops and profiling drivers hold the GPU lock. It is
explicitly excluded and archived under `preliminary/`. All three final profiles
were recollected sequentially after every benchmark, correctness and sanitizer
process completed. Scored comparisons in compare*.json preceded all profiling.

### Validation

- **45 native scored-shape reference comparisons:** 24 direct, 9 initial async,
  12 current async (some repeated configurations).
- Retained direct4/stage4 and async2/stage4 each passed **16/16** full reference
  tests, with five native cases and eleven production fallback cases apiece.
- **84 all-native fresh graph replays**, 42 per retained variant across seven
  nearby shapes: TP8/DP, pages1/64, scales 1 and 1/64, contexts 10–120K, soft/coherent/edge
  distributions, changed cache bytes/QKV/routing/lengths/positions/cache slots,
  including empty contexts.
- Both retained kernels isolated under Compute Sanitizer: **memcheck 0 errors;
  synccheck 0 errors; racecheck 0 hazards/errors/warnings**. Each tool ran 24
  native calls (eight variant/page/scale cases, three changed-QKV calls each).
  The sanitizer driver now also reduces FP32 partials independently in torch
  and checks final output against reference, rather than checking only finiteness.
- Deterministic source-generation scripts reproduce the archived variant files.
  All subprocesses completed; no experimental code was checked or scored.

### Judge progress, budget and next cycle

At **23:02 UTC**, exact-current-source check **01bc4ecc9c7047d9** had changed from
queued to **running**. At the 23:04 read it was still running. No result or profile
for that job yet; do not resubmit it. The older 40e1aa41415a4c87 result is resolved
in session 6 and cannot qualify this newer source for scoring.

Read Claude's completed main65 check **254b1d40063f429c**: **16/16**, **2.041433x**,
CI **[2.010811, 2.072054]**, no flags/error. Its JSON is saved in
`profiles/session7-main-check.json`. COORD.md now records a pending Claude scored
run for that same source (ID not recorded there), so refresh the actual board
before any Codex submission; don't assume the older 1.945619x remains the best.

Codex UTC-day usage remains **3/8 checks, 1/3 scored**. Session 7 used **0 checks,
0 scored**, no withdrawals. Shared account at 23:02 had **12/20 checks, 6/10
scored**, leaving 8/4. Last board read still **1.945619x**, rank **33**, job
**033117b46b6f42ba**. Session status/board/budget snapshots are in profiles/.

1. Resolve running **01bc4ecc9c7047d9** and read its passing profile. If it beats
   the then-current account best and budget permits, score the identical source.
   Existing production already has local full-reference/fresh/sanitizer coverage.
2. Fetch/merge main before new code; further source changes require a new exact-
   source check. Do not treat main's 254b result as a check of the native variant.
3. Close producer conversion as another measured dead end. More resident warps
   and fewer consumer barriers alone do not fix the native attention gap. Do not
   repeat its stage/warp sweeps, direct-register V, whole-cache packing or prior
   gather/merge families without a concrete new reduction in total work.
4. Keep the experimental branch separate. Production source is unchanged while
   its exact combined-code H200 confirmation is running.

Neither STOP file existed at the last read. Local work is complete; final judge
status and commit/coordination handoff follow below if they change while wrapping up.
