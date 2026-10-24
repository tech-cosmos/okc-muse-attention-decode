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

Final **23:06 UTC** read: **01bc4ecc9c7047d9 is still running**, so no new scored
run is justified yet. Account best remains **1.945619x**, now rank **34**; shared
usage remains **12 checks / 6 scored**, Codex **3 checks / 1 scored**. Main remains
65ff1ab and production SHA256 is unchanged. No STOP files, all local subprocesses
finished. Experiment branch is pushed at **f2e241e**, production notes/profiles at
**474b7c4** plus this final snapshot. Shared claim removed and result posted to
COORD.md. Next cycle starts by resolving the running check, without resubmitting.

## Session 8 — 2026-10-23, 23:07–23:17 UTC

### Outcome and production state

Resolved the current-source H200 check and submitted its exact source. A new
native register-pool experiment increased occupancy but lost on throughput, so
**no production kernel change** was made. The focused cycle closes with all local
work complete, without waiting idle through the scored job's queue.

`origin/main` remains **65ff1ab**, already merged. Production submission.py is
unchanged from **5e2441f/effc2bf**, SHA256
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
The existing small-full native reduction remains; no experimental native attention
is dispatched. Experiments are pushed on **codex-exp/cuda-register-pool** through
**21abcaf**, source/results commit **4923527**. Read
`experiments/register_pool/README.txt` on that branch for source and reproduction.

### Exact-source check and scored submission

**01bc4ecc9c7047d9 completed at 23:06 UTC** and was read at session start:
**16/16 passed, 2.009167x**, CI **[1.979029, 2.039304]**, no flags or rejections.
Read its passing profile; JSON/log/text are saved in `profiles/submission.01bc*`.
H200 times: decode1 full **14.953 us**, verify1 full **16.322 us**, full verify16/32
**155.649 / 305.298 us**. Native reduction profile durations are **1.8 / 2.4 us**.
PDL profile rows overlap/include waiting; do not sum them to estimate runtime.

At **23:08:05 UTC**, submitted **a48b8c849e53426b**, byte-identical production source.
The current board was **1.945619x**, rank **34**, so the exact-code/check-beats-board
condition was satisfied. Local full-reference/fresh/sanitizer validation already
covered this unchanged source in sessions4/5; it was not redundantly repeated.
Submission creation/status, check, board and budget snapshots are in
`profiles/session8-*.json`. Do not resubmit this source.

Main65's separate check254b was **2.041433x**, versus native **2.009167x**, with
overlapping confidence intervals. Native H200 decode1 full essentially ties
main (**14.953 vs 14.939 us**); verify1 full is slightly faster (**16.322 vs
16.548 us**). All ten unchanged shapes ran slower in the native check, so this
cross-run aggregate does not isolate a regression or benefit from the native
merge. Keep the prior same-process H100 comparisons as the stronger component
measurement. Claude's scored main run was still pending in COORD.md; refresh the
actual board next session.

### Register-pool experiment

Hypothesis: donate producer registers to consumer warpgroups, allowing three
resident native CTAs instead of two without shrinking the attention tile. Pad
producers to one complete four-warp group; adjust gather strides/barrier counts.
Preserve every key, FP8 cache/query precision, FP16 PV operands, FP32 partials,
new-token writes and fresh-input graph behavior. This stays in the native lane.

Two retained designs place repartition after the common prologue barrier:

- **late2:** initial80 registers, consumer96/96 + producer48; two blocks/SM.
- **late3:** initial56 registers, consumer72/72 + producer24; three blocks/SM.

Same-process H100 cold-L2 CUDA graph medians, empty replay subtraction, shared
GPU timing lock; sequential comparisons, not a balanced paired estimator:

| full verify | B1 | B16 | B32 |
|---|---:|---:|---:|
| production | 18.480 us | 155.552 us | 304.352 us |
| old native | 34.800 | 230.080 | 442.112 |
| static four-producer control, two blocks | 36.704 | 238.080 | 465.280 |
| late2 | 37.760 | 247.184 | 486.080 |
| late3 | 39.296 | 236.256 | 455.440 |

Both lose; no dispatch change or experimental judge check. After full correctness,
native/fallback full benches were **1.548x / 1.565x** against fixed SGLang/H200
reference times, not production regressions or judge results.

Placement matters: early24/72 and early40/64 hints were discarded by ptxas C7507,
leaving a spilled56-register kernel. Volatile query rematerialization did not
preserve those early hints. Early48/96 survives but heavily spills and takes
**514 / 1004 us** at B16/32. Moving24/72 after the common prologue makes real
USETMAXREG instructions survive. Stage4/page1 late3 still has104-byte compiler
spill-store/load counts; late2 has32 bytes each. Those are static compiler
counts, not measured dynamic traffic. SASS excerpts and all build logs archived.

Final isolated NCU confirms the intended mechanism: original/late2/late3 achieve
**29.02 / 35.60 / 47.12% occupancy**, issue **0.56 / 0.59 / 0.62** warps per scheduler
cycle, and take **8.24 / 9.57 / 12.13** warp cycles per issued instruction. Higher
occupancy is insufficient here. All benchmark/reference/fresh processes exited
before profiling; NCU waited for the sanitizer's GPU lock and then ran sequentially.
Do not treat NCU durations as graph timings.

Validation:

- **36** independent reference comparisons in the two timing drivers, including
  baseline/production/control repetitions (not 36 distinct native configurations).
- late2 and late3 each pass **16/16** full tests: five native, eleven fallback cases.
- **84 all-native fresh graph replays**, 42 per variant on seven nearby shapes;
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge inputs;
  changed cache/QKV/routes/lengths/positions/cache locations, including empty context.
- Each sanitizer runs 24 native calls across both variants. **Memcheck 0 errors;
  synccheck 0 errors; racecheck 0 hazards/errors/warnings**. Independent torch
  reduction of native FP32 partials is checked against reference, including caches.
- pool3/reload3 are compile-only, with no output-correctness or timing claim.
  Generation script reproduces the archived sources. All subprocesses completed.

### Budget and next cycle

Session8: **0 checks, 1 scored run**, no withdrawals. Codex UTC-day totals:
**3/8 checks, 2/3 scored**. Shared account at 23:16: **12/20 checks, 7/10 scored**, so
8 checks and 3 scored remain account-wide. Last account best is **1.945619x**,
rank 34. **a48b8c849e53426b remains queued** at the final read.

1. Read a48b8c849e53426b without resubmitting. Read its passing profile when ready;
   compare with the latest account best/Claude's pending main score.
2. Fetch/merge main before any new source work. The current source is already
   checked and submitted; changes require their own exact-source qualifying check.
3. Close this register-repartition family: the mechanism works, but increased
   occupancy plus the larger producer group/spills does not beat the old native
   kernel, much less production. Avoid more register/stage/grid sweeps absent a
   distinct reduction in work/dependencies. Earlier gather/conversion/packing
   families remain closed too.
4. Keep rejected experiments on their branch; end on clean, pushed `codex`.

Neither STOP file exists. Shared claim removal, results entry and final push are
part of this handoff. No outstanding local GPU work remains.

## Session 9 — 2026-10-23, 23:18–23:29 UTC

### Outcome and source state

Native shared-memory instruction reduction improves the old native attention
prototype by **8–9%** on large full-verify shapes, but it remains **32–36% slower
than production**. No production kernel change or new judge job. The focused
cycle ends with all local experiments/validation complete.

Experiment branch **codex-exp/cuda-matrix-loads**, pushed at **37a785b**. Start with
`experiments/matrix_loads/README.txt`; `forced.py` is the best retained native
variant. It is not imported or dispatched by production.

Merged **origin/main 468a53d** into codex as **e56d174**. Main changed notes and
profiles only. The add/add conflict in profile254b retained main's complete
report, which contains the same kernel measurements plus summary/shape timings.
Submission source remains byte-identical to **5e2441f/effc2bf**, SHA256
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No Triton code or plans were tuned. Existing native small-full reduction remains.

### Instruction changes and results

This cycle targets a concrete reduction in per-tile shared-memory instructions,
without changing the previous producer/consumer topology, stages4, split plan,
cache precision, FP8 query rounding, FP16 PV operands, FP32 partials or all-key
attention. New-token/cache writes and fresh-input graph behavior are preserved.

1. **x4 matrix loads:** replace two ldmatrix.x2.trans operations with one x4.
   Stage4/page1 SASS has **8 LDSM instead of 16**, with no new conversion/shuffle
   step. Runtime changes are marginal: B16/32 **229.0/438.3 us** versus old native
   **229.8/439.4**, production **155.0/303.9** in that comparison.
2. **First-write QK scores:** explicit output-only WGMMA plus scale-d=false.
   Baseline already emits a first QGMMA with RZ/!UPT. The full stage4/page1 SASS
   for baseline and this variant is byte-identical. Close this no-op direction.
3. **Vector V conversion:** each consumer loads eight contiguous FP8 bytes,
   converts all four pairs exactly to FP16, and stores a uint4. This replaces
   eight narrow shared loads with two 64-bit loads. nvcc scalarizes one of the
   two uint4 stores into four scalar stores; the other remains 128-bit.
4. **Forced vector stores:** inline st.shared.v4.u32 preserves both 128-bit stores.
   Combined with x4 matrix loads, this is the best new native building block.

Final same-process H100 cold-L2 graph medians, empty replay subtracted:

| full verify | production | old native | vector + x4 | explicit vector stores + x4 |
|---|---:|---:|---:|---:|
| B1 | 18.224 us | 34.640 us | 34.912 us | 34.944 us |
| B16 | 154.816 | 230.432 | 218.208 | 210.976 |
| B32 | 305.120 | 442.832 | 415.600 | 402.000 |

The driver holds the shared GPU lock through captures, reference checks and
timings. Sixty rounds rotate/reverse order, with within-round deltas archived.
This is **not** the judge's balanced adjacent-pair estimator; do not interpret
sub-percent differences as established wins. The native improvement and remaining
gap to production are much larger. Three comparison runs are archived separately.

Resources remain **94 registers/thread, 62,528 shared bytes, no spills**, two
CTAs/SM. Final native-only NCU baseline/vector+x4/forced occupancy is
**29.01/29.70/29.37%**, issue **0.56/0.53/0.55** warps per scheduler cycle, warp
cycles per issued instruction **8.22/8.91/8.56**. The improvement comes without
more occupancy or fewer FP8 conversions. Static SASS is archived; no dynamic
per-instruction counts are claimed. Optional PC sampling is absent, so no detailed
stall attribution is claimed. Profiles ran sequentially after all other GPU work;
do not use profiler durations as graph timings.

### Validation

- **39 variant/shape entries** across comparison drivers, including repeated
  controls; independent reference checks before and after graph capture.
- Retained vector+x4 and forced-store variants each pass **16/16** full tests:
  five native cases, eleven fallback cases each. Runtime validation is stages4;
  other compiled stages have no new runtime-validation claim.
- Full native/fallback benches after tests: **1.635x / 1.645x** respectively,
  against fixed SGLang/H200 reference numbers. These are experimental local
  aggregates, not production regressions or judge results.
- **84 all-native fresh graph replays**, 42 per variant across seven nearby
  shapes: TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge
  inputs, changed QKV/cache/routes/lengths/positions/cache slots, empty contexts.
- Native-only Compute Sanitizer: **memcheck 0 errors, synccheck 0 errors,
  racecheck 0 hazards/errors/warnings**. Each tool executes 24 calls across both
  variants, four page/scale cases and three changed-QKV calls each. Independent
  torch partial reduction/output transform is checked against reference/cache.
- Generator reproduces all six variants. All local subprocesses exited cleanly.
  Build/profile logs have trailing whitespace stripped only for git hygiene.

### Judge progress, budget and next cycle

Scored job **a48b8c849e53426b** was queued at session start and changed to
**running at the final 23:28 UTC read**. No result/profile yet. Do not resubmit it.
This is still the unchanged exact source of passing check01bc **2.009167x**.

Claude's scored **0a2eec0fdfad40af** has completed: **16/16, 2.020924x**,
CI **[1.990611,2.051238]**. Account best is now that result, **rank21**, rather
than the older 1.945619x. The current-source check is below this new board best;
it does not justify another score. The existing pending score was legitimately
submitted in session8 when the board was still 1.945619x.

Session9 uses **0 checks, 0 scored**, no withdrawals. Codex UTC-day totals remain
**3/8 checks, 2/3 scored**; shared account **12/20 checks, 7/10 scored**, leaving
8 checks/3 scored account-wide. Status/board/budget snapshots: profiles/session9-*.

1. Resolve running **a48b8c849e53426b** without resubmission. Read its passing
   profile when available, compare with latest board/main shape measurements.
2. Fetch/merge main before new source work. Further source changes require their
   own exact-source check before a scored run; preserve the daily budget rules.
3. If a distinct native architecture warrants more work, use `forced.py` as the
   better building block. Avoid further small stage/register/load-width sweeps.
   Vectorized V staging reduced per-tile work, but alone cannot close the gap.
   The x4-only change is marginal and first-write initialization is a compiler no-op.
4. Keep the rejected native attention experiment separate; production dispatch
   is unchanged. This session does not wait idle through the scored run.

Neither STOP file exists. Finish with the experiment and codex branches pushed,
the shared claim removed, and the worktree clean on codex.

## Session 10 — 2026-10-23, 23:30–23:40 UTC

### Outcome and production source

Resolved the previously pending scored run. Native FP16 warpgroup PV with a
swizzled value layout improves the retained native attention prototype by about
**2–3%** at large batches, but remains **29–33% slower than production**. No
production source change or new judge job. Experiments live on
**codex-exp/cuda-pv-warpgroup**, under `experiments/pv_warpgroup/README.txt`.
Experiment source/results are pushed at **6dcfd46**; final validation is below.

`origin/main` remains **468a53d**, already merged; fetched at session start and
after the comparisons. Production source is still exact **5e2441f/effc2bf**,
SHA256 `a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No Triton code/plans were tuned. Existing native small-full reduction remains.

### Completed H200 scored run

**a48b8c849e53426b** completed at **23:29:58 UTC**, read this session:
**16/16 passed, 2.011402x**, CI **[1.981231,2.041573]**, no flags or errors.
Read the full passing profile and saved text/JSON/log in `profiles/submission.a48*`.
This is the identical source of check01bc **2.009167x**, not a new submission.

H200 native-path decode1/verify1 full **14.922 / 16.670 us**; native reduction
profile durations **1.8 / 2.4 us**. Full verify16/32 **155.205 / 304.462 us**.
Main's scored0a2eec0f target times are **14.997 / 16.908 us**, but these are
independent judge runs. Overall account best remains main's **2.020924x**, rank21
at this session's first board read. Confidence intervals overlap; prior paired
H100 comparisons are stronger evidence for the small native merge benefit.
PDL profile durations overlap and include waits; do not sum them as latency.

### Native PV experiment

Base is session9 `forced.py`: paired-head full-row FP8 gathers, two producer
warps, two consumer warpgroups, mbarrier ring, vectorized FP8-to-FP16 V staging.
Use the same stages4/split heuristic, FP8 cache/query rounding, FP16 P/V operands,
FP32 accumulators/partials, all-key attention and new-token/cache behavior.

- **sync:** two register-input m64n64k16 FP16 WGMMA PV operations replace eight
  LDSM.x4 and sixteen warp HMMA. V is blocked N-major, initially unswizzled.
- **overlap:** alternate V buffers; defer PV completion to next QK wait. SASS
  confirms removal of the in-loop end-PV wait, but throughput gets worse.
- **swizzle:** N-major SW128 V storage,1024-byte alignment and matching matrix
  descriptor. Vector conversion/store widths are preserved. This removes most
  of the slowdown introduced by the unswizzled WGMMA layout.
- **swizzle_overlap:** same layout with deferred PV; loses to synchronous swizzle.
- **early_release:** publish the raw K/V ring slot after QK completion, before
  softmax/PV; remove the end consumer barrier. Separate converted V stays live
  until PV completion. Correct in initial tests but no timing gain.

Three same-process H100 cold-L2 graph comparisons, with empty replay subtraction
and shared GPU lock through captures, reference checks and timing. Sixty rounds
rotate/reverse order; this is not the judge's balanced adjacent-pair estimator.
Do not compare absolute values across runs to infer small effects.

| full verify | production | native vector baseline | unswizzled WGMMA | deferred unswizzled |
|---|---:|---:|---:|---:|
| B1 |18.464 us|35.440 us|36.592 us|36.720 us|
| B16|154.912|211.600|264.896|270.512|
| B32|304.880|402.032|515.248|525.392|

| full verify | production | native vector baseline | swizzled WGMMA | deferred swizzled |
|---|---:|---:|---:|---:|
| B1 |18.464 us|34.656 us|34.336 us|34.432 us|
| B16|156.160|212.528|207.136|210.336|
| B32|303.392|399.120|390.256|396.496|

| full verify | production | native vector baseline | swizzled WGMMA | early raw release |
|---|---:|---:|---:|---:|
| B1 |18.432 us|34.544 us|34.208 us|34.288 us|
| B16|154.928|211.056|205.552|206.272|
| B32|302.464|397.584|388.768|391.040|

Stage4/page1 resources: baseline94 registers/62528 shared bytes; sync94/61504;
overlap95/69696; swizzle92/61504; early_release92/61504. No spills; still two
resident CTAs/SM. Source generator, static SASS and operation counts archived.

### Validation and next work

- **36 variant/shape entries** across three comparison scripts, including repeated
  controls; independent reference checks before and after graph capture.
- Retained **swizzle and early_release each pass16/16** full-reference tests:
  five native full-verify cases and eleven production fallback cases each.
- Both full native/fallback benches report **1.657x** against fixed SGLang/H200
  numbers. These are experimental local aggregates, not production regressions.
- **84 all-native fresh graph replays**,42 per retained variant, across seven
  nearby shapes: TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge,
  changed QKV/cache/routes/lengths/positions/cache slots and empty contexts.
- Runtime validation is stages4. Discarded sync/overlap/swizzle_overlap have
  scored reference/capture comparisons only, not full-suite/fresh/sanitizer coverage.
- Generator reproduces all five sources. Duplicate initial build launcher was
  terminated before any GPU work; completed-build lock files were cleaned up.
  The retained comparison process completed normally, with no timing overlap.

1. No pending Codex judge jobs remain; do not resubmit resolved a48b8c849e53426b.
2. Fetch/merge main before further source changes. A new candidate needs its own
   exact-source check beating the then-current account best before scoring.
3. Keep swizzle.py as the best native building block. Close deferred PV and
   early-release variants; neither hides enough useful work to beat it.
4. A distinct future hypothesis is the untouched producer/raw-K QK shared layout:
   investigate bank conflicts first, then consider a matching K-major swizzle.
   Do not assume it wins; avoid further stage/register sweeps of this topology.

Session10 uses **0 checks,0 scored**, no withdrawals. Codex UTC-day totals remain
**3/8 checks,2/3 scored**. Shared initial usage **12/20 checks,7/10 scored**, leaving
8 checks/3 scored account-wide. No STOP file existed at the last read. Finish with
experiment/source reports pushed separately, clean codex, and claim removed.

Final validation/profile results: swizzle and early_release isolated native
memcheck **0 errors**, synccheck **0 errors**, racecheck **0 hazards/errors/warnings**.
Each tool checks24 native calls (two variants, four page/scale cases, three changed
QKV calls each), reducing FP32 partials independently in torch and checking output
and cache against reference. All local subprocesses have completed.

Isolated NCU baseline/sync/swizzle/early_release measured shared-store bank
conflicts **1.107M / 19.087M / 1.320M / 1.398M**; shared-load conflicts are about
**1.293M** for each. These are whole-kernel hardware counters, not per-instruction
attribution. They support the unswizzled-value store-conflict diagnosis. Achieved
occupancy **29.40 / 30.04 / 29.18 / 28.97%**, issued warps per scheduler cycle
**0.55 / 0.38 / 0.54 / 0.54**. Swizzling restores issue throughput, but does not
materially improve occupancy or close the gap to production. Profiles ran after
all correctness/timing/sanitizer work; profiler durations are not graph timings.

Final fetch found main **f65dfa4**, notes/profile scripts only, with unchanged
submission.py. Read its session6 notes: measured sliding dependent-memory trips
and unsuccessful early-load Triton experiments. A native first-gather/prologue
overlap is another future direction, but would need a new efficient attention
kernel rather than reusing the rejected session2 sliding prototype unchanged.
Final board/budget read: rank21, best2.020924x, account12 checks/7 scored;
Codex3 checks/2 scored. No new jobs or pending Codex jobs.

Final handoff: experiment **6dcfd46** pushed. Main **f65dfa4** merged into codex
as **23bf400**; only notes/profile scripts changed. No redundant production GPU
rerun was needed for this merge. Source SHA256 remains unchanged. Neither STOP
file exists, and all local subprocesses completed. Production notes/profiles are
committed and pushed, with claim removal/results posted to COORD.md.

## Session 11 — 2026-10-23, 23:41–23:51 UTC

### Outcome and source state

Native raw-cache layout changes improve the prior best native attention prototype
by **7–8%** on large full-verify shapes, but it remains **18–23% slower than
production**. No production kernel change or new judge job. Completed one focused
layout cycle with reference, fresh-graph, sanitizer and instruction-level profiling.

Experiment **codex-exp/cuda-key-layout**, pushed at **157c4d0**. Start with
`experiments/key_layout/README.txt`; **rawvk64.py** is the best retained building
block. The other retained structural variant is paired128.py. Neither is imported
or dispatched by production. The experiment includes reproducible source generation,
comparison/validation scripts, static SASS, compiler resources and complete profiles.

Fetched main at start and before finishing. Latest **origin/main 89f8df9** was
merged into codex as **fc28537**; only NOTES.md changed. Read its session7 results:
Triton V register prefetch, block-diagonal paired heads, q-in-loop sliding and
other sweeps lose. No redundant production GPU rerun was needed for this notes-only
merge. Production source remains exact **5e2441f/effc2bf**, SHA256
`a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No Triton code or plans were tuned. Existing native small-full reduction remains.

### Measured mechanism and implementations

Baseline is session10 swizzle.py: synchronous FP16 WGMMA PV with SW128 converted V,
paired FP8 gathers, two producers, two consumer warpgroups. Initial NCU
instruction-level SourceCounters exposed two independent shared-memory issues:

- Each of the two raw-V LDS.64 loads uses four wavefronts where two suffice.
  Together they have **1,292,576 excess shared wavefronts** at B16/S16.
- QK WGMMA itself has **zero excess wavefronts**. The producer K gather stores,
  rather than the QK consumer reads, motivate changing the raw-K layout. Aggregate
  cache-gather excess wavefronts are **12,597,624**, versus ideal **2,584,128**.

Variants preserve the same stages4/split rule, every-key attention, FP8 cache and
query rounding, FP16 P/V operands, FP32 partials, new-token/cache behavior and
fresh-input graph operation:

1. **rawv:** remove raw FP8 V row padding; retain allocation sizes and PV layout.
2. **key64:** K-major SW64 raw K; 512-byte alignment/stride across eight rows,
   matching descriptor, second K32 starts at +32. Padded raw V unchanged.
3. **rawvk64:** combine those changes; same allocation sizes as baseline.
4. **paired128:** both K heads in one SW128 tile, with 128-byte physical rows and
   per-head descriptor offsets. Separate compact V arrays reduce scratch space.

Same-process H100 cold-L2 CUDA graph medians, empty replay subtracted. Sixty
rounds rotate/reverse order and retain within-round deltas. This is not the
judge's balanced adjacent-pair estimator. GPU lock covers data generation,
reference checks, captures and timed loops. Do not infer small effects across runs.

| first comparison | B1 full verify | B16 | B32 |
|---|---:|---:|---:|
| production | 18.304 us | 155.056 | 306.512 |
| prior native | 34.768 | 205.616 | 394.096 |
| rawv | 33.920 | 197.744 | 377.088 |
| key64 | 33.472 | 195.168 | 371.152 |
| rawvk64 | 33.248 | 190.704 | 362.848 |

| second comparison | B1 full verify | B16 | B32 |
|---|---:|---:|---:|
| production | 18.432 us | 154.336 | 308.112 |
| prior native | 34.768 | 205.760 | 395.216 |
| rawvk64 | 32.640 | 190.384 | 362.688 |
| paired128 | 32.880 | 192.016 | 366.592 |

The paired layout reduces conflicts and scratch further but is slightly slower.
No experimental H200 check is justified while both still lose to production.

Stage4/page1 compiled registers: baseline92, rawv90, key6494, rawvk6489,
paired12888. cuobjdump shared allocations are **62,528 bytes** for the first four
and **54,336** for paired128. ptxas reports 1,024 fewer bytes (61,504/53,312);
this distinction is archived explicitly. Zero stack/local allocations and spills;
all remain at two resident CTAs/SM. Conversion/matrix/vector load-store instruction
counts do not change. This is a layout improvement, not an instruction-count claim.

### Validation and final profiles

- **27 variant/shape entries** across the two comparison drivers, including repeated
  controls; independent output/cache reference checks before and after capture.
- **rawvk64 and paired128 each pass 16/16** full tests: five native, eleven fallback.
- Their full native/fallback benches report **1.717x / 1.706x** against fixed
  SGLang/H200 times. These are experimental local aggregates, not production results.
- **84 all-native fresh graph replays**, 42 each across seven nearby cases:
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge data,
  changed cache/QKV/routes/lengths/positions/cache slots, including empty contexts.
- Isolated native **memcheck 0 errors; synccheck 0 errors; racecheck 0 hazards,
  0 errors, 0 warnings**. Each tool executes24 calls across both variants, four
  page/scale cases and three changed-QKV calls; torch independently reduces FP32
  partials and checks output/cache against reference. No Triton loader involved.
- rawv/key64 isolated-component controls have scored reference/capture comparisons
  only, not full-suite/fresh/sanitizer coverage. Runtime validation is stages4.
- Generator reproduces all four sources. All local subprocesses completed.

Final NCU hardware baseline/rawv/key64/rawvk64/paired128 load conflicts:
**1,292,621 / 31 / 1,292,613 / 39 / 25**; store conflicts:
**1,306,878 / 1,348,653 / 852,156 / 760,364 / 657,703**.
Occupancy stays around29%; issue per scheduler rises **0.54 -> 0.58/0.59**.
Source-attributed raw-V excess wavefronts fall to zero with compact V, while
aggregate gather excess falls to **2,584,128** with rawvk64 and **1,292,064**
with paired128. QK remains at zero excess. profile-summary.json distinguishes
source-attributed wavefronts from whole-kernel hardware conflict counters.

The diagnostic baseline source profile ran before experiments. All final profiles
ran sequentially after this session's correctness/timing/sanitizer work, under the
shared GPU lock. No profile duration is used as graph timing. Fewer conflicts alone
do not close the remaining gap, as the paired128 graph results demonstrate.

### Budget and next cycle

Session11: **0 checks, 0 scored**, no withdrawals. Codex UTC-day2026-10-23 totals
remain **3/8 checks, 2/3 scored**. Shared account at final read: **12/20 checks,
7/10 scored**, leaving8 checks/3 scored. Account best remains **2.020924x**, now
**rank22**. No pending Codex jobs. Snapshots: profiles/session11-*.json.

1. Fetch/merge main and read the shared board before further source work. Any new
   production source needs its own qualifying exact-source check before scoring.
2. Use rawvk64.py if pursuing another native architecture. Close further simple
   raw-layout sweeps; the best measured layout is still slower than production.
3. A distinct unimplemented dependency hypothesis: issue QK before independent V
   conversion/staging, then wait before consuming scores. Current code converts V,
   launches QK, and immediately waits. Verify async/generic ordering for new-token
   K and warpgroup convergence; this differs from rejected deferred PV/early raw
   release. No speedup is claimed for this idea.
4. Keep experiments separate and finish on clean, pushed codex. No production
   attention prototype is promoted. Neither STOP file exists. Claim removal and
   shared result posting are part of this handoff.

## Session 12 — 2026-10-23 23:52 to 2026-10-24 00:01 UTC

### Outcome and source state

Completed the session11 QK/V-overlap hypothesis. Native QK now overlaps V
conversion in generated SASS, but **does not improve large-batch latency**.
Production remains unchanged; no new judge jobs. Experiments are pushed on
**codex-exp/cuda-qk-overlap**, source/results **7393688**, final log cleanup
**79e2e25**. Start with `experiments/qk_overlap/README.txt` on that branch.
The experiment includes source generation, tests, graph comparisons, fresh-data
replays, sanitizers, SASS/resource reports and isolated NCU profiles.

Fetched main at start and before finishing; **origin/main 89f8df9** remains
already merged. Production submission.py remains exact **5e2441f/effc2bf**,
SHA256 `a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1`.
No Triton source/plans were tuned. Existing native small-full reduction stays.
No redundant production GPU rerun was needed for unchanged source.

### Native scheduling changes

Baseline is session11 rawvk64, with only build/module names changed. Same
stages4, split heuristic, paired FP8 gathers, query/cache precision, FP16 PV
operands, FP32 partials, all-key attention and new-token/cache behavior.

- **overlap:** keep K's proxy fence and pre-QK consumer barrier; issue/commit
  QK, convert/stage V, fence and synchronize V, then wait for QK scores.
- **latebar:** defer V's publication barrier until after softmax, just before PV.
- **nobar / latebar_nobar:** respective versions without the extra pre-QK
  consumer barrier. Each consumer still acquires the completed producer
  mbarrier and executes K's proxy fence; consumers do not write K. All V stores
  remain fenced and synchronized before PV; QK scores are untouched until wait.
  PV still completes before the raw-ring empty-slot release.

SASS confirms QGMMA before the V stores and QK DEPBAR after them, with some V
loads/conversions interleaved between QK instructions. This is real instruction
rescheduling, not a throughput win. Removing the pre-QK barrier removes its
SASS instructions, but an extra proxy fence/MEMBAR remains compared to baseline.
Conversion, vector-store and matrix-operation counts are unchanged. Compiled
registers baseline/overlap/latebar/nobar/latebar_nobar **89/91/93/91/93**;
cuobjdump shared bytes **62,528** (ptxas **61,504**), no spills, two CTAs/SM.

Final isolated same-process H100 cold-L2 CUDA graph medians, empty replay
subtracted, 60 rotating/reversing rounds and within-round deltas. This is not
the judge's balanced adjacent-pair estimator; small differences are not robust
speedup claims. GPU lock covers generation, reference checks, captures and timing.

| full verify | B1 | B16 | B32 |
|---|---:|---:|---:|
| production |18.176 us|154.304 us|303.856 us|
| prior native rawvk64 |33.072|190.240|358.608|
| overlap |32.640|193.504|366.128|
| latebar |32.960|195.920|370.144|
| nobar |33.200|191.152|360.576|
| latebar_nobar |33.280|193.904|367.408|

Nobar's within-round deltas against prior native are **+1.216 / +2.240 us** at
B16/B32. It recovers most of the extra-barrier loss but still trails production
by **19–24%**. The best native building block remains session11 rawvk64.

The first comparison showed the same large-batch regression for overlap/latebar.
The second exploratory comparison overlapped the validation driver's lifetime:
reference tests serialized on the GPU lock, but bench.py's unlocked warmups could
have overlapped its final shape. Do not use that second comparison as clean timing.
The final all-variant comparison above ran after all other local tests/benches/
sanitizers finished, before sequential profiles; it is the authoritative result.

### Validation and profiling

- All **four variants pass 16/16** full reference tests: five native full-verify
  cases, eleven production fallbacks each. Each suite was followed by a full
  native/fallback bench: **1.702 / 1.683 / 1.685 / 1.693x**, respectively, against
  fixed SGLang/H200 numbers. These are local experiments, not judge results or
  production regressions.
- Nobar and latebar_nobar each pass **42 all-native fresh graph replays** (84
  combined): seven nearby shapes, TP8/DP, pages1/64, scales1/1/64, contexts10–120K,
  soft/coherent/edge data, changed QKV/cache/routes/lengths/positions/cache slots,
  including empty contexts.
- Isolated native **memcheck 0 errors, synccheck 0 errors, racecheck 0 hazards /
  errors / warnings**. Each tool runs24 calls across those two variants, four
  page/scale cases and three changed-QKV calls; torch independently reduces
  FP32 partials and checks output/cache against reference. No Triton loader.
- Extra-barrier overlap/latebar have full-suite and scored reference/capture
  coverage, but no fresh-input/sanitizer claim. Runtime validation is stages4.
- Final comparison has18 variant/shape entries, checked before/after capture;
  first comparison12, exploratory second15. Generator reproduces all variants.
  All local subprocesses completed. Only trailing whitespace/blank lines were
  removed from generated logs for git hygiene.

Isolated NCU baseline/nobar/latebar_nobar occupancy **28.97/28.53/28.73%**;
issued warps per scheduler **0.58 for all three**. Shared-load conflicts
**19/25/21**, store conflicts **747579/766737/688001**. Source-attributed raw-V
and QK excess shared wavefronts stay zero; gather excess stays **2584128**.
The extra overlap does not improve useful occupancy or issue throughput. Profiles
run sequentially after final timing, under the shared GPU lock; no profiler
latency is used as a graph time. `summary.json` records the validation audit.

### Budget and next cycle

Session12 uses **0 checks, 0 scored**, no withdrawals; no pending Codex jobs.
UTC day **2026-10-23** closes at Codex **3/8 checks, 2/3 scored**, shared account
**12/20 checks, 7/10 scored**. Limits reset while wrapping up: final official
read on **2026-10-24** reports account **0/20 checks, 0/10 scored**; Codex starts
that day at **0/8 checks, 0/3 scored**. Account best remains **2.020924x, rank22**.
Snapshots are `profiles/session12-{board,budget}-{start,final}.json`.

1. Fetch/merge main and read COORD.md before source work; preserve ownership.
   Any new production source needs its own exact-source check beating the current
   account best before scoring. No jobs from this session need follow-up.
2. Close QK/V overlap and these barrier placements. Keep session11 rawvk64 as
   the native building block; more stage/register sweeps are not justified.
3. The remaining native direction with a concrete motivation is sliding first-
   gather/prologue overlap from main's session6 notes. It needs a new efficient
   attention design; the rejected session2 sliding prototype remains too slow.
   Claim only a distinct structural experiment, not another old-parameter sweep.
4. Finish on clean pushed codex; experiments stay separate. Neither STOP file
   exists. Shared claim removal and results posting complete the handoff.

## Session 13 — 2026-10-24, 00:02–00:18 UTC

### Outcome and source state

A new native **small-batch sliding-decode** path wins locally and is now on codex.
Final balanced cold-L2 graph comparison: **B1 ctx80K 4.864 -> 4.352 us**, paired
median delta **-0.432 us**, ratio **1.099x**. B2 improves **5.376 -> 5.024 us**,
paired delta **-0.304 us**. H200 confirmation remains pending; no scored run.

Native source commit **5641ff1** was fast-forwarded to codex. Experiment branch
**codex-exp/cuda-sliding-prologue**, source/reports **9c846a4**, is pushed. Start
with `experiments/sliding_prologue/README.txt` there. All experimental variants,
raw graph rounds, SASS, source profiles and validation drivers stay on that branch.
Production includes only the winning embedded CUDA and narrow dispatch.

Final fetch brought in main **57ad70c/908ec4f**: Claude's new full-verify rule,
4 programs/SM from six head-pairs up. Merged as **3fb391f** and read main session8.
No Triton kernel or plan was tuned by Codex. The merged submission separately
passes16/16 and benches **1.918x** vs fixed SGLang/H200 times; verify32 full is
**295.52 us**. Pre-merge integrated bench was1.916x, verify32 full304.94 us.
These are independent local runs, not paired aggregate speedup measurements.

Final submission.py SHA256:
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
Pre-main-merge source5641ff1 SHA256:
`1c7475a12eb2ca7dcb878b471c1b3c49e3b2d9bd5081e072a2914157d92a7d5b`.
The merge changes only the inherited full-verify plan; native code is unchanged.
Existing native small-full reduction remains. Experiments were not merged.

### Design and measurements

This follows main session6's first-gather/prologue observation, using a new
single-tile design rather than the slow session2 sliding kernel. One128-thread
CTA handles64 positions of one KV head. It issues cp.async K/V copies before
query normalization/RoPE. The new token is computed independently and inserted
into its natural tile after the zero-fill copies complete; no concurrent cache
write is read by attention. Every key in the window is included.

Four lanes per useful query head build FP8 E4M3 Q operands directly in registers;
the remaining WGMMA rows are zero. QK uses FP8 WGMMA, P/V use FP16 WGMMA, and
partials remain FP32. The inherited one-pass `_combine_hb` uses PDL and emits
normalized/gated BF16. Shape-only dispatch: window2048, L1, B*Hkv<=4, G8, D64,
page1/64, RoPE enabled. Larger batches and verify blocks retain main.

The shared-query control lost: B1 late6.112 / early5.888 vs production4.832 us.
Removing shared staging and padded-query work brought direct-Q late/early to
4.576/4.256 vs production4.672 in one same-process comparison. Early gather
ordering provides a further small gain; direct Q accounts for most of the
structural improvement. Direct early loses at B16/32 (11.57/19.26 vs10.40/15.76),
so it is not dispatched there. Explicit uint2 query loads do not improve the
B1 target; that control is archived and not promoted.

Final exact pre-merge integration, adjacent balanced AB/BA pairs,60 pairs/shape:

| sliding decode | parent | native | paired median delta |
|---|---:|---:|---:|
| B1 ctx80K page1 |4.864 us|4.352 us|-0.432 us|
| B2 ctx80K page1 |5.376|5.024|-0.304|
| B1 ctx4K page64 |4.936|4.512|-0.472|
| B2 ctx4K page64 |5.416|5.136|-0.264|
| B1 ctx1024 page64 |4.704|3.968|-0.760|

Per-variant medians need not subtract to the median of paired differences.
The earlier exploratory comparisons rotate/reverse all variants; they are not
this final balanced adjacent-pair estimator. All custom timings take the shared
GPU lock around generation, reference checks, capture and replay, flush512MB
before every replay, and subtract an empty replay. No profiler or own validation
process overlaps final graph timing. Bench.py's full benches retain its own
lock semantics (timed loops only).

Nearby comparisons cover B1/2, contexts10–120K, pages1/64, plus unchanged B3
fallback. Short contexts win too (ctx10 about1.1 us; ctx1024 about0.8 us).
All timing drivers check independent output/cache references before and after
capture. No H200 latency claim yet.

### Validation and mechanism

- Shared-Q/direct-Q late/early: four16/16 suites, three native cases each;
  vector controls late/early: two16/16 suites and scored comparisons.
- Integrated source5641ff1:16/16; final merged3fb391f independently16/16.
- Four shared/direct variants: **168 all-native fresh graph replays**,42 each,
  seven nearby shapes, TP8/DP, pages1/64, scales1/1/64, soft/coherent/edge inputs,
  changed QKV/cache/routes/lengths/positions/cache slots, including empty contexts.
- Integrated dispatch: **48 fresh graph replays**,42 native and6 DP fallback,
  including a120K context with1/64 scale. Native source is unchanged by the merge.
- Isolated native **memcheck0 errors; synccheck0 errors; racecheck0 hazards /
  errors / warnings**. Each sanitizer checks48 calls across four variants,
  two pages, two scales, three changing QKV calls. Torch independently reduces
  FP32 partials and checks output/cache; no Triton loader probes.
- Vector controls have no fresh/sanitizer claim. All relevant full benches ran.
- Three structural comparisons have12/20/20 variant-shape entries; nearby and
  integrated scans24 each; final balanced comparison10. Counts include controls.
- `audit.json` checks embedded production CUDA equals sanitized direct.cu.
  `profiles/session13-native-audit.json` retains that audit on codex. Its source
  hash refers to pre-main-merge5641ff1, as explicitly recorded there.

All native variants use74 registers, no stack/spills. Ptxas shared allocations
base20480 / direct16384 bytes; cuobjdump includes1024 additional bytes per CTA
(21504 /17408). Direct Q removes eight scalar LDS and six STS.U8 instruction
sites. Matrix counts remain two QGMMA and four HGMMA. SASS confirms direct
EARLY's first gather at0x6d0 before its first RSQ0x10d0; LATE gathers at0x1a00,
after RSQ0xb50. This is an actual schedule change.

Isolated NCU B1/S32 executed warp instructions: base early253768, direct
late163208, direct early164482. Source-attributed excess shared wavefronts
10248 ->4104. Profiling supports removing query staging; stall samples remain
spread over memory/barriers/constant and instruction loads. Profiler durations
are not graph times. Profiles ran after initial validation/comparisons and before
the final balanced graph comparison, sequentially under the shared GPU lock.

### Judge jobs, budget and next cycle

| job | source | state at handoff | qualification |
|---|---|---|---|
| check4d36508dda044b58 |5641ff1|queued|pre-main-merge only; cannot qualify latest source|
| check2776be989f794588 |3fb391f|pending|exact final source; resolve before scoring|

This cycle/day2026-10-24 Codex uses **2/8 checks,0/3 scored**, no withdrawals.
The official problem limit is now **10 checks/day**, not20; use the lower limit.
Known shared requests today: Claude1 + Codex2 = **3 checks**,0 scored, leaving
at most7 checks and10 scored account-wide. The first budget read just after the
second request lagged at2 while its status was `received`; the final official
read now confirms3/10 checks and0/10 scored. Snapshots: `profiles/session13-*.json`.
Account best remains **2.020924x**, rank23 at last read.

1. Resolve **both** jobs and read passing profiles. Do not resubmit them.
2. Only **2776be989f794588** can qualify the current source. Score only if it
   passes and beats the then-current board best, after refreshing the board.
   If further main code changes are merged, that combined source needs its own
   exact-source check. The older check is useful for isolating the native change.
3. Keep the native small-batch dispatch if H200 confirms the benefit. Do not
   extend to large batches or repeat the rejected shared-Q/vector-query sweeps.
4. A possible distinct next idea is a native L8 sliding single-tile kernel using
   all64 WGMMA rows. It needs causal handling of all eight own tokens and the
   union of their shifted windows; this session's L1 kernel must not simply be
   enabled for L8. No implementation or speedup is claimed for that idea.
5. The session11 rawvk64 full-attention building block remains slower than main;
   the recent QK/V overlap directions remain closed.

Experiment9c846a4 is pushed. Finish on clean pushed codex, remove the shared
claim and post results. All local GPU/build processes completed. Neither STOP
file exists. No scored run is authorized while the exact-source check is pending.

Final handoff: experiment branch is pushed through **0997cef** (final-merge
qualification notice; source/reports9c846a4). Production source/notes156eaa1
was pushed. Both judge jobs remain queued; source SHA256 is unchanged from
3fb391f. Official final budget confirms3/10 shared checks,0/10 scored. Shared
claim removed and results posted. No local subprocesses or STOP files remain.

## Session 14 — 2026-10-24, 00:19–00:33 UTC

### Outcome and source state

Implemented and rejected a distinct native **L8 sliding single-tile** design,
then two controls: vector query/RoPE loads, and preparation once per request/head
with dependent launch. All lose to production. **No production change, no new
judge jobs.** Experiment **fac5298** is pushed on `codex-exp/cuda-sliding-verify`, starting with
`experiments/sliding_verify/README.txt`. Sources, reproducible generators,
comparisons, full tests, fresh replays, sanitizers and SASS reports are archived.

Fetched main at start and during work; **origin/main908ec4f/57ad70c** is already
merged. Production remains exact **3fb391f**, SHA256
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
Existing native small-sliding L1 attention and small-full reduction stay.
No Triton source or plans were tuned. No redundant production-only suite was
needed for unchanged source; the experimental suites cover its fallback paths.

The CLI was absent from this session's PATH even after venv activation. Using
its existing `/home/daytona/.okc/bin/okc` installation resolved this; no install,
new credentials, login or rules acceptance was performed.

### Native L8 design and diagnosis

Each128-thread CTA handles64 positions of one KV head. All64 WGMMA rows now
contain useful queries: two tokens per warp, eight Q heads each. Per-row masks
cover its causal upper bound and individual2048-token lower bound. The union
has up to2055 positions, hence up to33 partials. Each new token is inserted into
its natural tile only after historical cp.async zero-fill copies finish; one
owner writes each new cache slot. Every key is attended. FP8 queries/cache,
FP16 PV operands, FP32 partials and normalized/gated BF16 output are preserved.

Both early/late gather orderings were tested. Vector controls use64-bit Q and
cos/sin loads for L8's fully occupied query tile. This does not reopen the prior
L1 vector sweep. The preparation-once variant writes fresh Q and new-KV scratch
in a separate timed native kernel. Dependent attention can begin its historical
gathers before waiting for preparation; coherent volatile Q loads follow
ACQBULK. The inherited PDL merge follows attention. All three kernels and
all input-derived work remain inside each timed call.

Independent balanced cold-L2 CUDA graph comparisons, EARLY versions, microseconds:

| L8 sliding | parent/native B1 | parent/native B16 | parent/native B32 |
|---|---:|---:|---:|
| fused scalar query |5.784/7.952|12.632/28.544|20.200/53.096|
| fused vector query |5.664/7.272|12.568/27.456|20.160/51.176|
| preparation once |5.760/8.256|12.704/29.728|20.080/52.160|

These are separate runs; compare each candidate with its paired parent.
Each entry has40 adjacent balanced AB/BA pairs. All custom drivers hold the
shared GPU lock over generation/reference/capture/replay, flush512MB before
each replay, and subtract empty replay cost. Nearby B2/page64/DP/scaled/edge/
coherent cases and B1 ctx10 lose too. Full-suite bench warmups did not overlap
these comparisons; all custom timing was completed before source profiling.

Offline merge-omission diagnostic at B1 ctx80K: parent full/attention-only
**5.728/4.112us**, native LATE **7.760/5.952**, native EARLY **8.016/6.368**.
Attention itself accounts for most of the regression. The33-to64 merge padding
is not the main explanation. Preparation-once removes repeated query work but
adds a launch, scratch traffic and dependency, with no end-to-end win. These
individual costs were not separately measured; avoid a stronger causal claim.

### Validation and mechanism

- All **six family/order combinations pass16/16**, three native L8 sliding
  cases and13 production fallbacks each; each suite followed by a full bench.
  Local native/fallback geomeans versus fixed SGLang/H200 numbers are scalar
  **1.729/1.723**, vector **1.755/1.735**, prepare-once **1.703/1.711**, LATE/EARLY.
  These are rejected experiments, not production regressions or judge results.
- Scalar and prepare-once each pass **84 all-native fresh graph replays**:
  42/order, seven shapes, TP8/DP, pages1/64, scales1/1/64, contexts10–120K,
  soft/coherent/edge data, changed QKV/cache/routes/lengths/positions/cache slots,
  including empty contexts. Combined168. Vector has no fresh/sanitizer claim.
- Three comparison drivers each have14 shape/order entries,42 total, with
  independent reference/cache checks before and after capture, including
  repeated production controls. Raw pairs and deltas are archived.
- Native-only sanitizers independently reduce FP32 partials in torch and check
  output/cache, without the Triton loader. Each tool/family executes24 calls:
  two orderings, two pages, two scales, three changing-QKV calls. Scalar and
  prepare-once both have **memcheck0 errors, synccheck0 errors, racecheck0
  hazards/errors/warnings**. No local processes remain.

SASS page1 LATE/EARLY registers: scalar79/74, vector80/80, prepare-once
attention75/75, preparation42. All zero stack/local allocations and spills.
Attention shared bytes17408 from cuobjdump,16384 from ptxas; preparation0.
Vector query loads replace32 scalar BF16 plus32 scalar float sites with24
64-bit sites. Matrix operations remain2 QGMMA and4 HGMMA. Preparation-once
attention has758 static instructions versus scalar1481/1457, no RSQ, and
Q loads after ACQBULK; EARLY historical gathers precede ACQBULK. A separate
preparation kernel is added, so this is not a total instruction-speedup claim.

### Judge jobs, budget, and next cycle

Both session13 jobs were still queued at the latest read during local work:
**4d36508dda044b58** (older5641ff1) and **2776be989f794588** (exact current3fb391f).
Neither was resubmitted. Main's checkfdccc1c6fa704926 had moved to running.
Only2776 can qualify production for scoring; the older check can isolate the
L1 native change but cannot qualify the merged current source. No passing
profiles for these jobs exist yet. Final status/budget updates follow below.

This cycle **0 checks,0 scored**, no withdrawals. Codex2026-10-24 remains
**2/8 checks,0/3 scored**. Account start **3/10 checks,0/10 scored** (official
check cap remains10, not20). Board start: best **2.020924x**, rank24.

1. Resolve both pending Codex checks and read their passing profiles. Do not
   resubmit. Score exact current source only if2776 passes and beats the refreshed
   account best, within the official remaining budget. New main source changes
   require their own exact-source qualification.
2. Close this L8 single-tile/direct-query/preparation family; no native L8 path
   is promoted. The L1 win does not extend by filling all64 WGMMA rows.
3. Small merge changes alone cannot bridge the measured attention gap. More
   native work needs a distinct measured mechanism; do not repeat old vector,
   stage/register/gather/pipeline sweeps. The session11 full rawvk64 building
   block remains slower than main and its QK/V-overlap variants remain closed.
4. Fetch/merge main and read COORD.md before further source work; finish on
   clean pushed codex. Experimental sources and results remain separate.

Final source profiles: native LATE/EARLY execute315056/311872 warp instructions;
prepare-once attention192998 plus preparation5104. Source-attributed excess
shared wavefronts4104/4556/4564/0, respectively. Long-scoreboard stalls remain
distributed. Profiles ran sequentially under the shared GPU lock after timing/
full benches and serialized against sanitizers. No profiler duration is used
as graph timing. The audit reproduces sources and verifies all claimed logs.

Final fetch brought **origin/main8b0ac0c**, notes only, documenting main check
fdccc1c6:16/16,2.013x, verify32full295.4us. It does not qualify Codex's source.
It is below the account best, so main did not score it. No source change and
no redundant production test needed when merging these notes.

Final00:31–00:32UTC snapshots: both Codex checks remain **queued**. Official
account usage **3/10 checks,0/10 scored**, Codex **2/8,0/3**, best2.020924x,
rank24. No new job, withdrawal, or STOP file. Do not wait idle or resubmit;
carry these exact two pending IDs into the next cycle.

Final handoff **00:33UTC**: experiment **fac5298** pushed; main8b0ac0c merged
as **01f390e**; production notes/audit **8a93715** pushed. Submission hash remains
unchanged. All local build/GPU/profile processes exited. Neither STOP file
exists. Shared claim removed, own pending-job lines refreshed, and results
posted to COORD.md. Finish on clean pushed codex; no experimental import added.

## Session 15 — 2026-10-24, 00:33–00:43 UTC

### Outcome and source state

Rejected native **L1 sliding last-CTA completion/reduction** on the current fast
single-tile attention kernel. Separating partial loads from reduction recovers
much of the first prototype's loss, but still loses to production. No production
source change, new judge check, or scored request. Experiment **338c474** pushed
on **codex-exp/cuda-sliding-completion**, with sources, reproducible generators,
raw pairs, tests, fresh replays, sanitizers, disassembly and audit. Start with
`experiments/sliding_completion/README.txt` on that branch.

Main **8b0ac0c** was already merged and remained current at the final fetch.
Production still matches source **3fb391f**, SHA256
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
No Triton kernel or plan was tuned. Native small-sliding L1 attention and the
small-full reduction remain as before. No redundant unchanged-production suite
was run; both experimental suites exercise its fallback paths.

### Completion design and measured result

The session13 attention arithmetic is preserved. Each block writes FP32 partials;
all writers threadfence, then synchronize with a leader that publishes through a
device-scoped acquire/release atomic. The last CTA per request/KV head merges all
splits, normalizes/gates and stores BF16 output. No CTA spins. Partial loads are
volatile coherent inline PTX, after last-CTA detection. The scheduling counter
returns to zero before kernel completion; no input-derived state is cached.

Two forms: load/weight/reduce each split immediately, and separate source loops
for loading partials and reducing them. SASS confirms the second issues **nine**
128-bit partial loads before the first reduction FFMA versus **one** originally;
it then interleaves further loads and math, rather than retaining all32 splits
at once. The separate-merge control preserves the original PDL attention/merge.
All native experimental dispatches are broad L1 sliding G8/D64/page1 or64 for
measurement, not proposed production dispatches.

Balanced adjacent cold-L2 CUDA graph pairs, H100 microseconds:

| sliding L1 | parent / initial fused | parent / prefetched fused |
|---|---:|---:|
| B1 ctx80K page1 |4.384 /10.432|4.416 /6.576|
| B2 ctx80K page1 |4.992 /10.944|5.040 /6.848|
| B1 ctx4K page64 |4.432 /10.240|4.224 /6.528|
| B2 ctx1024 page64 |4.144 /8.832|4.320 /6.016|
| B1 ctx10 page1 |3.776 /7.008|3.872 /5.856|
| B16 ctx80K page1 |10.416 /18.512|10.336 /14.992|
| B32 ctx80K page1 |15.712 /27.264|15.792 /23.936|

Compare each candidate with its own paired parent. Each comparison has40 AB/BA
pairs per entry. B1/B2 prefetched paired deltas are **+2.128/+1.840us**; they need
not equal differences of individual medians. B3 ctx4K fusion loses as well.
Separate full experimental benches report **1.716x/1.803x**, initial/prefetched,
versus fixed SGLang/H200 times. These are rejected experiments, not production
regressions or judge results.

Offline decomposition,60 pairs per entry: B1 attention-only **3.232us**, attention
with completion tracking but no reduction **4.608**, complete fused **6.624**.
B2 attention/completion-only **3.488/4.864**, attention/full-fused **3.424/6.944**.
The completion-only paired increase is **1.376/1.408us** before final reduction.
This measures fences, synchronization, atomic publication and last-CTA detection/
reset together, not isolated atomic latency. The first two diagnostic modes
intentionally omit the final output and CANNOT implement kernel_fn or be judged.
They are confined to an offline diagnostic; torch independently reduces and
reference-checks their partials/cache before/after capture and after timing.

All custom GPU drivers hold the shared lock across generation, reference checks,
capture and replay; flush512MB before every replay and subtract an empty graph.
Full benches started after comparisons/tests/sanitizers finished and retain
bench.py's own timing-loop lock. No NCU run was necessary. SASS/resource inspection
does not launch kernels. All local subprocesses exited.

### Validation and source evidence

- Initial and prefetched each **16/16**, three native cases and13 fallbacks.
- Each **48 all-native fresh graph replays**, eight shapes, six seeds each,
  including empty contexts. TP8/DP, pages1/64, scales1/1/64, contexts10–120K,
  soft/coherent/edge inputs; QKV/cache/routes/lengths/positions/cache slots change.
- Each comparison has16 entries (eight candidates/eight controls), independent
  reference/cache checks before and after capture,32 entries total.
- Each candidate/tool checks24 isolated native calls: two contexts257/2048,
  two pages, two scales, three fresh-QKV calls. Both candidates: **memcheck0,
  synccheck0, racecheck0 hazards/errors/warnings**. No Triton loader involved.
- Fresh and sanitizer drivers explicitly assert counters return to zero.
- Generated source reproduction and reported logs are checked by `audit.py`;
  production copy is `profiles/session15-native-audit.json`.

Initial/prefetched fused use **74/78 registers**; both have **18468 shared bytes**
from cuobjdump (17444 ptxas), zero stack/local memory and no spills. Controls are
74regs/17408 shared bytes (16384 ptxas). Page1 SASS instruction counts:
control1049, initial fused1813, prefetched1725. Attention's two QGMMA/four HGMMA
remain; fused barrier instruction sites rise from3 to6. These are static counts,
not an occupancy or measured instruction-throughput claim. Archived generated
logs only had trailing whitespace/blank EOF lines trimmed for git hygiene.

### Judge jobs, budget and next cycle

At the final **00:41UTC** read both session13 jobs remain **queued**:
**4d36508dda044b58** (older5641ff1) and **2776be989f794588** (exact current3fb391f).
Neither was resubmitted; no passing profile exists yet. Only2776 can qualify
current source for scoring. Refresh the board before acting on it.

This cycle **0 checks,0 scored**, no withdrawals. Codex2026-10-24 remains
**2/8 checks,0/3 scored**; official shared usage **3/10 checks,0/10 scored**.
The actual account check cap is10/day, not20. Board best **2.020924x**, rank26.
Snapshots are `profiles/session15-*.json`. Neither STOP file exists.

1. Resolve both existing checks without resubmitting. Read passing profiles;
   score exact current source only if2776 beats the then-current account best.
   Any further main source merge needs its own combined-source qualification.
2. Close last-CTA fusion on this L1 attention design. Improved partial-load
   overlap cannot overcome completion/reduction costs here. Do not repeat
   last-CTA, register, stage or small-merge sweeps without a new mechanism.
3. One incidental nearby lead: at B3 ctx4K page1, separate-merge native control
   runs **5.280/5.184us** versus parent **6.208/6.240** in the two comparisons.
   This is outside production's B*Hkv<=4 dispatch and was not swept or promoted.
   A future small-batch boundary survey could investigate it, respecting broad
   nearby shapes; it is not a scored-load speedup. B16/32 native still loses.
4. Fetch/merge main and read COORD.md before source work. Session14's L8 family
   and prior full-attention gather/overlap sweeps remain closed. Finish on clean
   pushed codex, with experiments separate and shared claim removed.

Final handoff **00:43UTC**: experiment **338c474** and production notes/audit
**0811206** pushed. Shared claim removed, own pending-job lines refreshed, result
posted; other agent's claim preserved. All local build/GPU/sanitizer/benchmark
processes completed. Final worktree is clean on codex; neither STOP file exists.
No experimental import or source change was added to production.

## Session 16 — 2026-10-24, 00:43–00:54 UTC

### Outcome and source state

Confirmed the **nearby small-batch native sliding dispatch opportunity** and
rejected two padded-query softmax reductions. No production source change and
no new judge jobs. Experiment **ad3bbef** is pushed on
**codex-exp/cuda-sliding-boundary**, starting with
`experiments/sliding_boundary/README.txt`. It contains the proposed dispatch patch,
reproducible CUDA controls, graph pairs, tests, fresh replays, sanitizer reports,
SASS and audit. Production audit copy: `profiles/session16-native-audit.json`.

Main **8b0ac0c** was already merged and remained current at final fetch. Production
still matches **3fb391f**, SHA256
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
No Triton source or plans were tuned. The expanded dispatch is deliberately held
as a separate patch so pending check2776 still qualifies the exact production
source. It changes no scored dispatch; do not spend a check solely to remeasure
unchanged scored paths. Bundle it with a future qualified production change.

### Resolved older H200 check

**4d36508dda044b58**, source5641ff1, completed around00:48UTC and was read:
**16/16,2.027312x**, CI **[1.996903,2.067969]**, no flags/rejections/errors.
Read the full passing profile; text/JSON/log saved as `profiles/submission.4d36508*`.

H200 decode1 sliding **5.995698us**, native attention profile **3.0us**, inherited
merge **1.5us**. Small-full decode1/verify1 **15.362767/16.329053us**, native
reduction profile **1.9/2.4us**. Full verify16/32 **155.573262/306.313008us**.
PDL profile rows include waiting/overlap; do not add them to infer graph latency.
This supports retaining the native small-sliding path, alongside session13's
paired H100 evidence. Cross-run differences alone do not isolate its benefit.

The aggregate is slightly above account best2.020924x with overlapping intervals,
but it is the **pre-main57ad70c** source. It CANNOT qualify the newer merged
production source for scoring. Exact-current check **2776be989f794588** remains
**queued at00:53UTC**. No resubmission and no scored request were made.

### Dispatch boundary: a nearby-shape win

The unchanged native kernel was forced on L1 sliding G8/D64/page1 or64 shapes,
then compared with production using60 adjacent balanced AB/BA graph pairs/entry.
The broad candidate wrapper uses the conservative shape-only rule **B*Hkv<=16**
(up from4), retaining every other predicate. Proposed patch is archived, not applied.

Second-seed confirmation,80K/page1,H100 microseconds:

| batch | production | original native | paired delta |
|---|---:|---:|---:|
|3|6.336|5.312|-1.088|
|4|6.432|5.536|-0.912|
|5|6.976|6.144|-0.768|
|6|7.296|6.624|-0.688|
|8|7.808|7.424|-0.384|

Initial scan agrees. Nearby contexts10–120K, pages1/64, mixed/agentic lengths,
soft/coherent/edge inputs and1/64 scales also favor the extension. DP1 (=16 total
KV heads) wins in the measured short/long cases; DP2 at long context loses.
Final edge scan includes B7/8/9/10, both pages. B9 page1 is marginal (-0.128us),
B10 page1 loses (+0.336us); page64 can win beyond8. Keep the conservative16-head
boundary rather than fit those small differences. Initial B12/16/32 native loses.
This is a useful generalization, **not a scored-load or H200 speedup claim**.

### Padded-query work: no scored improvement

The current native WGMMA tile has64 query rows but only8 real output heads.
`pruned.cu` computes softmax/probability packing only for warp zero's8 useful
rows, with the other probability operands zeroed; all128 threads reconverge for
collective PV. `half.cu` drops each warp's second8 padded rows without the
warp-zero branch. Actual queries, all keys, cache writes, FP8 query/cache
precision, FP16 PV operands and FP32 partials are preserved.

Balanced comparisons against the ORIGINAL NATIVE path, separate runs:

| shape | base/pruned | base/half |
|---|---:|---:|
|B1 80K page1|4.464/4.432|4.272/4.272|
|B2 80K page1|4.976/4.800|4.928/4.848|
|B16 80K page1|11.536/11.920|11.520/11.200|
|B32 80K page1|19.312/20.720|19.312/19.232|

B1 paired deltas **-0.064/0.000us**, insufficient for promotion. Pruned's warp
branch increases registers **74->106**; half retains74. Both have17408 shared
bytes (cuobjdump),16384(ptxas), zero stack/local allocations and spills. Static
page1/EARLY SASS instruction sites base/pruned/half **1049/1185/945**, EX2 sites
**32/32/16**, F2FP **52/44/44**, matrix counts unchanged2QGMMA/4HGMMA. Pruned has
additional branches/duplicated code. These are static counts, not executed warp
instructions; no NCU or measured occupancy claim. Less padded-row arithmetic
alone does not deliver a convincing latency win here. Close these controls.

### Validation and reproducibility

- Pruned and half each **16/16**,3 native/13 production fallback tests. Each
  followed by full broad-native bench, **1.886x/1.892x** vs fixed SGLang-H200
  numbers. These are rejected experiments, not production regressions.
- Proposed boundary wrapper independently **16/16**,3 native/13 fallback;
  full bench **1.923x**. Its scored dispatch/arithmetic is unchanged, so this
  aggregate is not evidence of a scored speedup over the earlier1.918x run.
- Base/pruned/half each **60 all-native fresh graph replays**,180 total,10 cases
  and6 seeds: changed QKV/cache/routes/lengths/positions/cache slots, including
  empty contexts; TP8/DP, pages1/64, scales1/1/64, contexts10–120K.
- Pruned/half each **memcheck0 errors, synccheck0 errors, racecheck0 hazards/
  errors/warnings**. Each candidate/tool checks12 native calls (B6,ctx2048,
  two pages,two scales,three changed-QKV calls); torch independently reduces
  partials and checks output/cache. No Triton loader. Only EARLY=True is runtime
  validated in this session; compiled EARLY=False remains unused.
- **104 comparison entries**,60 pairs each, including repeated controls;
  references before capture, after replay, and after timing. Every custom GPU
  driver locks generation/reference/capture/replay,512MB flush each replay,
  empty replay subtraction. Full benches ran without our other GPU work and
  retain bench.py's timing-loop lock. All local processes have completed.
- `audit.py` reproduces generated CUDA, checks all completed logs/counts and
  confirms proposed patch applies. Only generated text whitespace was normalized.
  Source and full reports remain on the experiment branch.

### Budget and next cycle

This cycle **0 checks,0 scored**, no withdrawals. Codex2026-10-24 stays
**2/8 checks,0/3 scored**. Final official account usage **3/10 checks,0/10 scored**;
actual check cap remains10/day, not20. Board best **2.020924x**, rank26.
Snapshots: `profiles/session16-*`. Neither STOP file exists.

1. Resolve **2776be989f794588** without resubmitting. Read its passing profile;
   score exact current3fb391f only if it beats the then-current account best.
   Older4d36508 is resolved; do not poll it again or use it to qualify latest code.
2. Preserve the nearby-boundary patch for the next actual source update; it has
   strong local evidence and a broad shape rule, but changes no scored load.
   Any integrated source must receive its own qualifying check before scoring.
3. Do not repeat padded-row softmax, last-CTA fusion, L8 single-tile/preparation,
   register/stage/merge sweeps or the rejected full native pipelines. Further
   native work needs a distinct measured mechanism. Read Claude's active Gluon
   claim/results before choosing overlapping architecture work.
4. Finish on clean pushed codex, with experiment ad3bbef separate, shared claim
   removed and results posted. No experimental import or production edit remains.

## Session 17 — 2026-10-24, 00:55–01:07 UTC

### Outcome and source state

Resolved exact-production check **2776be989f794588**: **16/16, 2.052839x**,
CI **[2.022047, 2.083632]**, no flags/rejections/errors. Submitted that identical
production source as **71e2eaa7b52c407f**, still **queued at01:06UTC**. Do not
resubmit it. No new check and no experimental kernel was sent to the judge.

Rejected a distinct **warp-sized single-tile sliding L1 attention** design,
including cooperative gathers, hybrid WGMMA-QK/warp-PV, and matching V-layout
controls. No production source change. Experiments and complete reproducible
reports are pushed as **44bda60** on **codex-exp/cuda-sliding-warp**; start with
`experiments/sliding_warp/README.txt`. Production evidence copy:
`profiles/session17-native-audit.json`.

Merged notes-only **origin/main e4e037c** as **bf3ca62** and read its session9
Gluon results. No Triton kernel or plan was tuned. Production still matches
**3fb391f**, SHA256
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
Session16's nearby B*Hkv<=16 patch remains separate; no source update was needed.

### Exact-source qualification and scored request

Check2776 moved from queued to running around00:58UTC and completed around01:04.
Read its entire passing profile; text/JSON/log saved as `profiles/submission.2776*`.
H200 decode1 sliding **5.805422us**, decode1/verify1 full **14.490277/16.131548us**,
verify16/32 full **154.760676/295.909571us**. Native attention profile **3.0us**,
inherited sliding merge **1.5us**; native small-full reduction **1.8/2.4us**.
PDL profile rows include waiting/overlap, so do not sum them as graph latency.

Refreshed board immediately before scoring: best **2.020924x**, rank26; official
account usage3/10 checks,0/10 scored. The check exceeds that best, satisfying the
same-source submission rule. Its interval overlaps the old best's interval;
this is a qualifying check, not a claim of a statistically established new rank.
Local exact-production16/16 and bench coverage is already recorded in session13;
source hash was reverified before scoring, with no redundant unchanged suite.

Scored request71e2 was created around01:04UTC from exact3fb391f/48f0775 source.
No experimental import or dispatch was added. Final01:06 snapshot remains queued,
board best2.020924x/rank27. Resolve the score next cycle, read its profile if it
passes, and update the shared board. The old check4d36508 is already resolved;
check2776 is now resolved too. Neither should be polled or submitted again.

### Warp-sized design and measured rejection

The production native tile has64 query rows but only8 real heads for L1. New
`mma.sync` variants use16 rows (8 real,8 padding). They retain the single64-key
tile, early cp.async, query normalization/RoPE and FP8 rounding, new FP8 K/V
insertion, FP16 P/V, FP32 partials and unchanged production PDL merge. Every
key and value is attended; only noncausal/out-of-window padding is masked.

- **warp:** one32-thread CTA handles gathers and both matrix products.
- **cooperative:**128 threads gather/convert; warp zero handles both products.
- **hybrid:** retain production's true FP8 QK WGMMA; warp zero handles softmax/PV.
- Each also has a **swizzle** control preserving the parent's V shared mapping
  and matching ldmatrix addresses, removing an avoidable linear-layout change.

Balanced adjacent AB/BA CUDA-graph pairs,60 pairs per shape, H100 microseconds:

| variant | B1 parent/candidate | B1 paired delta | B2 parent/candidate |
|---|---:|---:|---:|
| warp |4.480/6.432|+1.920|5.120/6.752|
| cooperative |4.448/5.472|+1.056|5.072/6.096|
| hybrid |4.448/4.944|+0.480|5.040/5.552|
| warp, swizzled V |4.608/6.384|+1.744|4.912/6.496|
| cooperative, swizzled V |4.448/5.360|+0.992|5.088/5.952|
| hybrid, swizzled V |4.640/5.088|+0.288|5.056/5.600|

Each row has its own paired parent; do not compare small differences across runs.
The swizzled hybrid has small incidental B8/DP1 wins but still loses B1/B2. No
shape rule is promoted. B16/32 controls force the original native kernel, which
itself loses to production there; parity against those controls is insufficient.
The broad experimental wrapper is not a proposed production dispatch.

Static SASS explains a disadvantage: warp/cooperative FP8 mma.sync lowers to
FP16 HMMA on this Hopper build. They have64 HMMA +32 LDSM sites versus baseline
2 QGMMA +4 HGMMA. Hybrid keeps2 QGMMA but replaces4 HGMMA with32 HMMA +32 LDSM.
Restoring V swizzling does not recover B1 latency. More instructions and changed
gather topology are consistent with the losses, not an isolated causal proof.
No NCU, achieved-occupancy, or dynamic instruction-count claim was made.

Base/warp/cooperative/hybrid page1-EARLY registers **74/99/64/69**; corresponding
swizzled controls **100/64/71**. All use17408 shared bytes (cuobjdump),16384(ptxas),
zero stack/local allocation and no spills. Generated CUDA and SASS are archived.

### Validation, budget, and next cycle

- All **six variants16/16**, each3 native/13 production fallbacks, followed by
  the full benchmark. Aggregates are rejected experimental results, not production
  regressions; exact values are in the audit and per-variant logs.
- **360 all-native fresh graph replays**,60 per variant on10 shapes: changed
  QKV/cache/routes/lengths/positions/cache slots, including empty contexts;
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge inputs.
- Original warp/cooperative/hybrid each **memcheck0, synccheck0, racecheck0
  hazards/errors/warnings**. Each tool/variant checks12 native calls and
  independently reduces FP32 partials with torch, checking output/cache.
  Swizzled controls have no separate sanitizer claim. EARLY=True is runtime
  validated; EARLY=False is compiled but unused in this cycle.
- **120 comparison entries**, including controls,60 pairs/shape. Independent
  output/cache checks before capture, after replay and after timing. Custom
  drivers lock generation/reference/capture/replay and flush512MB each replay;
  all timings subtract an empty replay. Full benches retain bench.py's timing
  lock. No profiler ran. All local build/GPU/sanitizer processes have completed.
- audit.py reproduces generated sources, verifies baseline equals production's
  embedded native CUDA, and checks complete logs. Only generated report whitespace
  was normalized for git hygiene.

Session **0 checks,1 scored**, no withdrawals. Codex2026-10-24 totals
**2/8 checks,1/3 scored**. Official account **3/10 checks,1/10 scored**, leaving
7 checks/9 scored account-wide. The official check cap remains10, not20.

1. Resolve **71e2eaa7b52c407f** without resubmission; it is the exact checked source.
2. Close warp-sized MMA and its value-layout controls for this L1 design. Do not
   repeat padded-row, fusion, warp-count or small-merge sweeps absent a distinct
   measured mechanism. Main's Gluon full-attention work remains Claude's lane.
3. Keep session16's nearby-boundary patch for a future actual source update,
   requiring a new exact-source qualifying check if integrated.
4. End on clean pushed codex; experimental branch remains separate, shared claim
   removed/results recorded. Neither STOP file exists. Do not wait idle for the
   queued scored job; carry its exact ID into the next session.

## Session 18 — 2026-10-24, 01:08–01:16 UTC

### Outcome and source state

Rejected native **single-tile sliding L1 QK/V scheduling** changes: actual
instruction overlap, but no useful scored-shape gain. No production change and
no new judge requests. Experiment **5b842c0** is pushed on
**codex-exp/cuda-sliding-qk-overlap**, starting at
`experiments/sliding_qk_overlap/README.txt`. Reproducible source, graph pairs,
validation, sanitizer logs and SASS are archived there. Audit copy on production:
`profiles/session18-native-audit.json`.

Fetched main at start and end; **origin/main e4e037c** was already merged.
Read Claude's active Gluon warp-specialization claim and stayed in the native
lane. Production remains exact **3fb391f**, SHA256
`16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e`.
No Triton source/plans or production dispatch changed. Session16's nearby
B*Hkv<=16 patch remains separate and unintegrated.

### Mechanism and measured rejection

The native sliding kernel originally converts/stages V, issues QK and waits,
then computes softmax/PV. This session tests that single-tile latency path,
not session12's rejected full-attention producer/consumer loop:

- **overlap:** issue/commit QK before V conversion; fence/publish V, then wait.
- **late:** wait for QK and compute softmax before converting/publishing V.
- **pinned:** overlap with explicit post-wait compiler liveness for register Q
  operands. Page1/EARLY SASS instruction lines are identical to overlap.

New-token K stores have an async-proxy fence before the existing CTA barrier;
V retains its own fence/barrier. Matrix operations remain converged, scores
are compiler-pinned until completion, all keys/cache writes are preserved,
and the production PDL merge is unchanged. FP8 queries/cache, FP16 P/V,
FP32 partials and BF16 output precision remain unchanged.

Balanced adjacent AB/BA graph pairs,60 pairs/shape,H100 microseconds:

| variant | B1 parent/candidate | B1 paired delta | B2 parent/candidate | B2 delta |
|---|---:|---:|---:|---:|
| overlap |4.464/4.480|0.000|5.024/4.992|-0.048|
| late |4.320/4.320|0.000|5.088/5.056|-0.016|
| pinned |4.384/4.384|-0.016|5.024/4.960|-0.096|

Each row has its own paired control; small cross-run differences are not gains.
Nearby B3/8 and short B1 can regress. B16/32 comparisons force the old native
path, which already loses to production there; tiny improvements over that
control do not justify dispatch. No candidate was promoted or sent to H200.

Disassembly confirms QGMMA before V's four STS.128 operations and QK wait after
those stores for overlap/pinned. Late's QK wait precedes V stores; ptxas overlaps
some conversion with softmax. Baseline/candidates all use74 registers,
17408 shared bytes (cuobjdump;16384 ptxas), zero stack/local memory or spills.
Page1/EARLY static instructions1049->1045; matrix counts2QGMMA/4HGMMA,
32EX2 and3CTA barrier sites stay unchanged. No NCU, achieved occupancy,
dynamic-instruction or isolated matrix-latency claim is made.

### Validation and handoff

- All three variants **16/16**, each3 native/13 production fallback tests,
  followed by full benches. Broad-native aggregates **1.909/1.899/1.895x**
  (overlap/late/pinned) are experiments, not production regressions.
- Pinned/late each **60 all-native fresh graph replays**,120 total,10 shapes
  each; changed QKV/cache/routes/lengths/positions/cache slots and empty contexts.
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge inputs.
- Pinned/late each isolated **memcheck0 errors, synccheck0 errors,
  racecheck0 hazards/errors/warnings**. Each tool/variant checks12 native calls
  at B6ctx2048, two pages/scales and three changed-QKV calls; torch independently
  reduces partials and checks output/cache. No Triton loader. Overlap has no
  separate fresh/sanitizer claim. Only EARLY=True is runtime-validated here.
- **60 comparison entries**,30 candidate/control pairs of shapes,60 AB/BA
  rounds per pair; independent reference/cache checks before capture, after
  replay and after timing. Custom drivers lock generation/reference/capture/
  replay and flush512MB before every timed replay, subtracting empty graphs.
  Full benches run sequentially with our other GPU work and retain bench.py's
  timing-loop lock. No profiler ran. All local processes completed.
- Audit reproduces sources, verifies unchanged production hash, checks complete
  logs and SASS ordering. Generated reports only had whitespace/EOF cleanup.

Scored **71e2eaa7b52c407f remains queued at01:16UTC**. It is session17's
exact checked production source; no resubmission and no passing profile yet.
Do not poll the already resolved checks2776/4d36508 again. Current account
best remains **2.020924x**, rank27. Session **0 checks,0 scored**, no withdrawals;
Codex2026-10-24 remains **2/8 checks,1/3 scored**. Official shared account
**3/10 checks,1/10 scored**; the problem's actual check cap is10/day, not20.
Snapshots are `profiles/session18-*`. Neither STOP file exists.

Next cycle: resolve71e2 without resubmitting; read its passing profile if ready.
Fetch main/read COORD before source work. Close this L1 scheduling family along
with the earlier full QK/V overlap, padded-row, warp-MMA, last-CTA and small-merge
sweeps. Further native work needs a distinct, profiled bottleneck; moving these
same instructions again is not justified. Preserve session16's nearby-boundary
patch for a future substantive production update requiring its own exact-source
check. End this cycle on clean pushed codex, with experiment separate and shared
claim removed/results recorded; do not wait idle for the scored queue.

## Session 19 — 2026-10-24, 01:17–01:33 UTC

### Outcome and source state

Resolved scored **71e2eaa7b52c407f**: **16/16, 2.057970x**, CI
**[2.027101, 2.088840]**, no flags/rejections/errors. New account best,
**rank23**. This is session17's already-submitted exact3fb391f source; no
resubmission. Read the entire passing profile, saved under `profiles/submission.71e2*`.
The interval overlaps the previous best; do not attribute the entire aggregate
increase to native code or claim a statistically established rank improvement.

Merged **origin/main a82d996** as **d08ef14**: Claude's warp-specialized Gluon
full-verify kernel, dispatched above16 request/head pairs. Then integrated the
previously validated session16 **B*Hkv<=16** native sliding L1 boundary extension
as **45c56bc**. Native CUDA arithmetic and the small-full native merge are unchanged.
No Triton/Gluon code or plans were tuned. The boundary patch changes nearby shapes,
not a scored shape. No rejected prologue-load variant was integrated.

Final production SHA256:
`e9a877984ef70a7067cce7ed85b8e573c95edc693b661b6e7ed16fa99136f349`.
Sent this exact source for **check b4853b2accdc4b50**, still **queued** at01:31UTC.
Only this new check can qualify the new source for scoring. Resolved checks2776/
4d36508 and scored71e2 cannot qualify it. Do not resubmit any of those jobs.

### Combined-source validation and measured improvement

- Local full suite **16/16**, then full bench **1.944x** against the fixed
  SGLang/H200 numbers. Verify32 full **275.92us**. This local aggregate is not
  an H200 speedup estimate or a paired aggregate comparison.
- **24 fresh-input CUDA graph replays** over8 shapes,3 seeds each, with changed
  QKV/cache/routes/lengths/positions/cache slots and empty contexts. Nine native
  sliding calls,12 Gluon full-verify calls,3 existing Triton fallback calls.
  Full cases include B17/B24/B32 TP8 and B3 DP, pages1/64, scales1/1/64,
  contexts3000–80000; sliding cases B3/B8/DP1. All independent reference/cache
  checks pass. These extra full cases explicitly exercise the new Gluon path.
- Same-process balanced adjacent AB/BA CUDA graph comparisons against the old
  production source,60 pairs/shape, independent reference/cache before capture,
  after replay, and after timing:

| shape | previous production us | combined us | paired delta us |
|---|---:|---:|---:|
| B8 sliding L1 |7.696|7.424|-0.256|
| B16 full L8 (unchanged path control) |150.528|149.488|-1.024|
| B17 full L8 |160.416|165.344|+4.960|
| B24 full L8 |216.656|212.224|-4.144|
| B32 full L8 |289.744|269.616|-19.952|

Main's B32 improvement reproduces. The **B17 ~3% regression** is recorded in our
COORD line for Claude to review; we did not tune its owned dispatch. The small
unchanged B16 difference is timing variation. The nearby native B8 win agrees
with session16's broader evidence. No blanket nearby-shape speedup claim.

`profiles/session19-integrated-{test,bench,fresh,pairs}*` holds results and raw
pairs. Drivers and old-source snapshot are on the experiment branch described
below. Custom drivers lock generation/reference/capture/replay, flush512MB before
each timed graph, and subtract an empty replay. Full bench retains bench.py's
own timing lock; our other GPU work had completed first. No new integrated
sanitizer claim: native implementation is unchanged, while Gluon belongs to main.

### Native independent-load experiment: rejected

Experiment **51229d6**, pushed on **codex-exp/cuda-sliding-loads**, starts at
`experiments/sliding_loads/README.txt`. Baseline SourceCounters identifies a
page-result long-scoreboard wait before query loads. The two page indices
already load together, so batching them would not remove a serial round trip.

- **qpre:** preload raw BF16 Q with inline loads before gather; normalize later.
- **plain:** same source order with ordinary C++ loads.
- **rope:** also preload Q's eight cosine/sine pairs.
- **ordered:** volatile Q and page loads enforce Q-before-page issue order.

qpre/plain/rope can still calculate cache addresses from page results ahead of
Q loads in SASS. The ordered control actually emits Q loads first, but adds
STRONG.SYS semantics; it does not isolate ordering from memory semantics.
Every key, new-token/cache write, query/cache precision, FP16 PV, FP32 partial,
BF16 output and existing PDL merge are preserved.

Initial60 adjacent balanced pairs/shape,seed103,H100 microseconds:

| variant | B1 base/candidate | B1 paired delta | B2 paired delta |
|---|---:|---:|---:|
| qpre |4.240/4.256|0.000|-0.096|
| plain |4.320/4.336|0.000|-0.192|
| rope |4.352/4.160|-0.192|-0.080|
| ordered |4.352/4.816|+0.512|+0.544|

Rope's second-seed120-pair confirmation gives B1 **4.480/4.320us** but median
paired delta only **-0.048us**; nearby B2ctx1024/page64 regresses **+0.496us**.
No dependable scored gain justifies promotion or an experimental H200 check.
B16/32 forced-native controls already lose to production; no broad dispatch claim.
Do not infer paired savings by subtracting separately computed medians.

All variants compile to74 registers,17408 shared bytes (cuobjdump),16384(ptxas),
zero stack/local memory/spills. Matrix sites remain2QGMMA/4HGMMA,3CTA barriers.
SourceCounters executed warp instructions baseline/qpre/rope/ordered:
**164482/169922/167820/171714**; shared excess wavefronts **4104** in each.
Samples stay spread over memory/constant/instruction/barrier waits. No achieved
occupancy claim, and no use of NCU duration as graph timing.

Validation: all four **16/16** (3 native/13 fallback), each followed by full
bench; **240 all-native fresh graph replays**,60/variant over10 cases, including
TP8/DP, pages1/64, scales1/1/64, short/long/empty contexts and fresh metadata.
Rope/ordered each isolated **memcheck0, synccheck0, racecheck0 hazards/errors/
warnings**;12 native calls per tool/variant, independent torch reduction plus
output/cache reference. qpre/plain have no separate sanitizer claim. Runtime
coverage is EARLY=True; EARLY=False is compiled only. There are50 candidate/
control shape comparisons:40 with60 pairs and10 confirmation entries with120.
All custom GPU drivers hold the shared lock throughout; final profiles follow
all these comparisons and correctness work. All local processes have completed.

`profiles/session19-audit.py` reproduces generated variants from the archived
generator, confirms unchanged embedded native CUDA, and audits all pass counts,
sanitizer summaries, raw-pair counts and final source hash. Reports:
`profiles/session19-{native,validation}-audit.json`. Only generated report text
whitespace was normalized. Experiments remain separate from production imports.

### H200 profile, budget, and next cycle

Scored71e2 H200 decode1 sliding **5.73us**, decode1/verify1 full **14.42/16.14us**;
verify16/32 full **154.11/294.22us**. Native attention profile **3.0us**, sliding
merge **1.4us**; native small-full reduction **1.8/2.4us**. PDL rows include waits
and overlap; do not sum them to infer graph latency. This scored profile predates
the Gluon merge and boundary extension.

Session **1 check,0 scored**, no withdrawals. Codex2026-10-24 totals
**3/8 checks,1/3 scored**. Official shared account **5/10 checks,1/10 scored**,
leaving5 checks/9 scored. Actual check cap is10/day, not20. Current best
**2.057970x**,rank23. Neither STOP file exists. CLI is
`/home/daytona/.okc/bin/okc` (not on this shell's PATH; venv activation alone
also does not add it).

1. Resolve **b4853b2accdc4b50** without resubmitting; read its passing profile.
   Score exact45c56bc only if that check beats the then-current account best.
   Earlier scored71e2 is resolved and is the new best; do not poll it again.
2. Fetch main/read COORD. Claude now owns the new Gluon producer/consumer path
   as well as Triton. Our B17 regression evidence is recorded for its review.
3. Close the native Q/RoPE preloading family along with recent padded-row,
   warp-MMA, QK/V scheduling, last-CTA and small-merge sweeps. Future native work
   needs a distinct measured mechanism, not further placement permutations.
4. Session16's boundary patch is now integrated and validated; do not reapply it.
   Keep experiment51229d6 separate. End on clean pushed codex, shared claim removed
   and result posted; do not wait idle for the new check queue.

## Session 20 — 2026-10-24, 01:33–01:41 UTC

### Outcome and source state

Completed a focused native **FP32 partial-store vectorization** cycle. The
explicit PTX version halves store instructions and removes excess theoretical
sectors, but its end-to-end gains are small and uneven. **Not promoted; no new
judge requests.** Experiment **1c2c2b3** is pushed on
**codex-exp/cuda-sliding-stores**, starting at
`experiments/sliding_stores/README.txt`. Sources, reproducible generators,
machine code, graph pairs, reference tests and profiles are archived there.
Production evidence copy: `profiles/session20-native-audit.json`.

Fetched main at start and end; **origin/main a82d996** remains already merged.
Read Claude's active Gluon claim and stayed in the native lane. Production is
unchanged exact **45c56bc**, including session19's Gluon integration and native
B*Hkv<=16 boundary. SHA256:
`e9a877984ef70a7067cce7ed85b8e573c95edc693b661b6e7ed16fa99136f349`.
No Triton/Gluon code, plan, merge or production dispatch changed. No redundant
unchanged-production suite was needed; experimental suites cover fallbacks.

### Store mechanism and measured limits

Baseline source profiling shows 18 scalar FP32 epilogue store sites with excess
theoretical sectors, but few stall samples. Adjacent values already have aligned
addresses, so this cycle changes only their store width:

- **vector:** C++ float2 assignments. nvcc scalarizes accumulator stores again;
  only max/sum combines. SASS:16 scalar stores plus one64-bit store.
- **explicit:** inline PTX st.global.v2.f32. SASS:nine64-bit stores as intended.

Attention, every key/value, new cache entries, FP8 query/cache precision,
FP16 P/V, FP32 partials, BF16 output and PDL ordering are unchanged. Registers
remain74, shared allocation17408 bytes(cuobjdump)/16384(ptxas), zero stack/local
memory and spills. Both retain2QGMMA,4HGMMA,3CTA barrier sites. Audit static
counts include padding NOPs; do not compare with old differently filtered counts.

Balanced adjacent AB/BA cold-L2 graph pairs,60 rounds/shape,H100 microseconds:

| variant | B1 parent/candidate | B1 paired delta | B2 paired delta | B8 paired delta |
|---|---:|---:|---:|---:|
| vector |4.384/4.352|-0.016|-0.160|+0.112|
| explicit |4.288/4.192|-0.096|-0.080|+0.048|

These initial B1 comparisons use **agentic lengths**, not the exact fixed-length
scored B1. Explicit's second-seed120-pair confirmation gives B1 paired
**-0.048us**, B2 **-0.096us**, B3 **+0.032us**, B8 **+0.064us**. The separate
fixed-length B1 survey,120 pairs/shape, gives:

| context | parent/candidate us | paired delta us |
|---|---:|---:|
|40000|4.480/4.480|-0.048|
|80000 (scored shape)|4.544/4.448|-0.096|
|120000|4.512/4.576|0.000|

Paired deltas need not equal differences between independently computed medians.
The scored-shape saving is small, nearby shapes are mixed, and this does not
justify another H200 check. B16/32 forced-native improvements cannot justify a
dispatch: their original native controls already lose to production Triton.
Close this store-width family rather than tune more epilogue permutations.

Isolated SourceCounters verifies partial-store executed warp instructions
**1152->576**, theoretical L2 sectors **8448->4224**, excessive theoretical
sectors **4224->0**. Total executed warp instructions fall only
**164482->163394**, shared excess wavefronts stay4104. These are source metrics,
not measured HBM bytes; no occupancy or isolated store-latency claim. Profiler
duration is not graph timing. This is a real mechanism with little total payoff.

### Validation and reproducibility

- Both variants **16/16**,3 native/13 production fallbacks, then full benches.
  Broad-native aggregates **1.917/1.913x** are experiments, not production
  regressions or H200 estimates.
- **120 all-native fresh graph replays**,60/variant on10 cases, changing QKV,
  caches, routing, lengths, positions and cache slots, including empty contexts.
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge inputs.
- Explicit only: isolated **memcheck0 errors, synccheck0 errors, racecheck0
  hazards/errors/warnings**. Each tool checks12 native calls across two pages,
  two scales and three changed-QKV calls; independent torch partial reduction
  plus output/cache reference. No separate vector sanitizer claim.
- **33 candidate/control shape pairs**:20 initial at60 rounds each,10 second-seed
  and3 fixed-length at120 each. Independent reference/cache before capture,
  after replay and after timing. Custom drivers lock generation/reference/
  capture/replay, flush512MB before every replay and subtract empty replay.
  Full benches retain bench.py's timing-loop lock and had no concurrent own GPU
  work. Fixed-length comparisons and profiles serialize through the shared lock.
- Runtime coverage is EARLY=True; EARLY=False is compiled only. Audit reproduces
  generated sources, verifies native baseline equals production, checks complete
  pass/pair counts and sanitizer summaries. All local processes have completed.

Setup: the first production import was canceled before GPU work to avoid an
existing extension-cache lock. Builds moved to an isolated repo
TORCH_EXTENSIONS_DIR. Its old shell's delayed `Terminated` line appears in the
otherwise complete vector test log; this is not a failed kernel test. No shared
extension cache lock was removed. Generated reports had whitespace cleanup only.

### Judge state, budget and next cycle

**b4853b2accdc4b50 remains queued at01:40UTC.** No passing profile exists yet,
and no resubmission was made. Only this check can qualify exact45c56bc for
scoring; earlier2776/4d36508/71e2 are resolved and cannot qualify newer source.
Current account best remains **2.057970x,rank24**. Session **0 checks,0 scored**,
no withdrawals. Codex2026-10-24 totals **3/8 checks,1/3 scored**; official shared
account **5/10 checks,1/10 scored**, leaving5 checks/9 scored. Actual check cap
is10/day. Snapshots: `profiles/session20-*`. Neither STOP file exists.

1. Resolve **b4853b2accdc4b50** without resubmitting, and read its passing profile.
   Score exact45c56bc only if it beats the then-current account best within budget.
2. Fetch main/read COORD before further work. Claude owns Triton/Gluon; its B17
   full-verify regression observation from session19 remains available to review.
3. Native partial-store width now joins the closed small L1 optimization families.
   Avoid more placement/width sweeps; future native work needs a distinct measured
   bottleneck large enough to matter. This experiment stays separate from production.
4. End on clean pushed codex, remove shared claim and post result. Do not wait idle
   for the queued check. Session16 boundary patch is already integrated.

## Session 21 — 2026-10-24, 01:42–01:51 UTC

### Production integration and judge qualification

Merged **origin/main0a22b94** as **9064c53**, then main's notes-only **3daf4a4**.
This adds Claude's producer-side FP8-to-FP16 V conversion on the existing Gluon
full-verify path. Native sliding attention, its B*Hkv<=16 boundary, and the native
small-full reduction are unchanged. No Triton/Gluon code or plan was tuned.

Production SHA256:
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.
New exact-source **check24414e478fa345a2** was requested at01:49UTC and remains
**queued**. Older **b4853b2accdc4b50** also remains queued; it qualifies only
exact45c56bc (SHAe9a877984ef7), not the newly merged source. Neither was
resubmitted. No passing profile exists for either yet. Resolved earlier jobs
2776/4d36508/71e2 need no further polling.

Current account best remains **2.057970x,rank24**, scored71e2. Only score the
exact source of a passing check that beats the then-current account best;
refresh the board and budget immediately before doing so. Do not substitute
the newer submission.py for old checkb485's source.

### Integrated validation and paired timings

- Local full suite **16/16**, followed by full bench **1.956x** against the
  fixed SGLang/H200 reference numbers; verify32full **272.13us**. This local
  aggregate is not an H200 estimate or a paired aggregate comparison.
- **24 fresh-input graph replays** over8shapes,3seeds each:9native sliding,
  12new Gluon full-verify,3existing Triton fallback calls. Changed QKV/cache/
  routes/lengths/positions/cache locations and empty contexts; TP8/DP, pages1/64,
  scales1/1/64, contexts2049–80000. Output/cache references all pass. In particular,
  these extra full cases exercise the new Gluon path beyond the standard suite.
- Independent same-process adjacent balanced AB/BA graph comparisons against
  exact45c56bc,60pairs/shape, with reference/cache checks before capture,
  after initial replay and after timing:

| shape | previous production us | merged us | paired delta us |
|---|---:|---:|---:|
| B8 sliding L1, unchanged control |7.408|7.456|-0.016|
| B16 full L8, unchanged control |149.744|149.536|-0.256|
| B17 full L8 |164.528|161.344|-3.360|
| B24 full L8 |212.256|208.224|-4.016|
| B32 full L8 |268.944|267.056|-2.144|

The latest main improvement reproduces, including a partial recovery of the
previously noted B17 regression. Do not infer equality with the older pre-Gluon
B17 source from measurements across sessions. Small unchanged-control differences
are timing variation. Pair deltas need not equal differences of separate medians.
Artifacts are `profiles/session21-integrated-*`; reproducing drivers and the
exact old-source snapshot are on the experiment branch below. No new integrated
sanitizer claim: the native implementation is unchanged; Gluon belongs to main.

### Native paired FP8 conversions: rejected

Experiment **c4f224a** is pushed on **codex-exp/cuda-sliding-fp8pairs**, starting
at `experiments/sliding_fp8pairs/README.txt`. It is not imported by production.
The prior SourceCounters/SASS has scalar FP32-to-E4M3 instructions with unused
packed operands. This cycle pairs conversions without changing rounding,
saturation, precision, attention arithmetic, all-key coverage or synchronization:

- **query:** two adjacent rotated Q values per packed conversion.
- **both:** additionally pair new K/V values, retaining their original stores.

Page1/EARLY SASS base/query/both FP32-to-FP8 sites **20/12/10**, allF2FP
**52/44/42**, PRMT **8/0/0**, LOP3 **49/29/31**. All retain74registers,
17408shared bytes(cuobjdump)/16384(ptxas), zero stack/local memory/spills,
2QGMMA/4HGMMA and3CTA barrier sites.

Balanced adjacent AB/BA cold-L2 graph pairs,60rounds/shape,H100 microseconds:

| variant | fixed B1ctx80K parent/candidate | B1 paired delta | B2 delta | B8 delta |
|---|---:|---:|---:|---:|
| query |4.336/4.352|+0.016|-0.080|+0.112|
| both |4.304/4.320|-0.016|+0.016|+0.096|

Each row has its own control. Initial B1 is the exact scored fixed-length shape;
nearby B3,short/page64/DP cases are mixed. There is no target gain to confirm,
so no additional sweeps, dispatch change or experimental H200 request.

Isolated SourceCounters base/query/both FP32-to-FP8 executed warp instructions
**1032/520/516**, byte-permutation instructions **512/0/0**, total instructions
**164482/161922/162182**. Shared excess wavefronts remain4104. This removes
about1.5% of total instructions without improving the complete graph. No achieved
occupancy or isolated conversion-latency claim. NCU durations are not graph times.

Validation and reproducibility:

- Both variants **16/16** (3native,13production fallback), then full broad-native
  benches **1.911/1.910x**. Those are experimental aggregates, not production
  regressions or H200 estimates.
- Each **60all-native fresh graph replays**,120total over10cases, changed
  QKV/cache/routes/lengths/positions/cache slots, including empty contexts.
  TP8/DP, pages1/64, scales1/1/64, contexts10–120K, soft/coherent/edge data.
- **24bit-exact comparisons** of FP32 partials and new cache bytes against
  native baseline:12case/seed combinations per variant.
- Both-conversion variant only: isolated **memcheck0errors,synccheck0errors,
  racecheck0hazards/errors/warnings**. Each tool checks12native calls across
  pages1/64,two scales,three changed-QKV calls; torch independently reduces
  partials and checks output/cache. No separate query-only sanitizer claim.
- Native comparisons cover16candidate/control shape pairs,60rounds each.
  All custom drivers lock generation/reference/capture/replay, flush512MB
  before every timed replay and subtract an empty replay. Full benches run
  sequentially with our other GPU work and retain bench.py's timing-loop lock.
  Profiling ran after all timing, reference and sanitizer work. All local
  processes completed. Runtime coverage is EARLY=True; EARLY=False is compiled.
- Audit regenerates variants, verifies native baseline equals production and
  checks all pass/pair/sanitizer counts. Production evidence copy is
  `profiles/session21-native-audit.json`. Only generated report whitespace was
  normalized; the audit was rerun successfully afterward.

### Budget and next cycle

Session **1check,0scored**, no withdrawals. Codex2026-10-24 totals **4/8checks,
1/3scored**. Final official shared usage is **6/10checks,1/10scored**, leaving4checks/9scored.
The first post-request read lagged at5checks; the final read now includes the
sixth request. Actual check cap
is10/day, not20. Snapshots are `profiles/session21-*`. Neither STOP file exists.

1. Resolve **24414e478fa345a2** (latest source) and **b4853b2accdc4b50** (older
   source), without resubmitting. Read passing profiles; only the latest check
   can qualify current submission.py. Preserve source identity when scoring.
2. Fetch main/read COORD before work. Latest merged main is3daf4a4; native
   kernels and the nearby boundary remain unchanged. Claude owns Triton/Gluon.
3. Close paired FP8 conversions and the earlier small-L1 micro-optimization
   families. The removal of roughly1.5% of instructions did not move latency.
   A more substantial future native direction could be the full-verify merge
   in Gluon's new three-kernel PDL chain. First isolate its actual critical-path
   contribution: main's long merge profile rows include waiting and are not
   evidence that all that time is reducible. Do not reopen old merge sweeps
   without that measurement and a distinct structural change.
4. Keep experimentc4f224a separate; end on clean pushed codex, shared claim
   removed and result posted. Do not wait idle for queued checks.

## Session 22 — 2026-10-24, 01:51–02:03 UTC

### Outcome and production state

Measured the full-verify merge's actual critical-path contribution, then rejected
native grouped-row and split-striped replacements. **No production source change,
no new check.** Experiment **8cd75fa** is pushed on
**codex-exp/cuda-verify-merge**, starting at `experiments/verify_merge/README.txt`.
Sources, graph pairs, diagnostic controls, tests, fresh replays, sanitizers,
compiler resources, SASS and reproducible audit are archived there. Production
evidence copy: `profiles/session22-native-audit.json`.

Fetched main at start/end; **origin/main3daf4a4** remains already merged.
Production is unchanged SHA256
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.
No Triton/Gluon source, plans, merge or production dispatch was tuned. Native
small sliding attention, its B*Hkv<=16 boundary, and small-full reduction stay.

### Resolved H200 check and exact-source scored request

Older **b4853b2accdc4b50**, exact45c56bc/SHAe9a877984ef7, completed around02:00UTC:
**16/16,2.069426x**, CI **[2.038385,2.110797]**, no flags/rejections/errors.
Read every passing profile; complete JSON/log/readable profile saved under
`profiles/submission.b4853b2accdc4b50.*`. H200 decode1 sliding **5.980us**,
decode1/verify1 full **14.813/16.451us**, verify16/32 full **155.387/265.111us**.
The old Gluon profile has attention256.3us, merge120.4us, new tokens3.4us,
GPU busy267.4us; those overlapping rows include waits and cannot be summed as
end-to-end latency. This check predates producer-side V conversion in current code.

Refreshed board **2.057970x,rank25**, official budget6/10checks,1/10scored.
The passing check exceeds that best, within overlapping confidence intervals.
Submitted **78b6db684e514dd6** around02:01UTC from the **exact older source**,
exported with `git show 45c56bc:submission.py` into
`profiles/session22-qualified/submission.py`. SHA256 verified as
`e9a877984ef70a7067cce7ed85b8e573c95edc693b661b6e7ed16fa99136f349`;
new scored job's content_hash matches b485's exactly. Existing session19 local
16/16, bench, fresh and paired validation applies to this unchanged source;
no redundant suite or rollback of current production was needed.

**78b6db684e514dd6 remains queued.** Current-source **24414e478fa345a2** also
remains queued. Only24414 can qualify current SHA6b171beb8ed5 for scoring;
b485 is now resolved and must not be polled/resubmitted. Earlier2776/71e2 etc
are resolved too. No new experimental source went to the judge.

### Merge decomposition and native result

Same-process balanced adjacent AB/BA cold-L2 CUDA-graph pairs,80rounds/entry,
seed103. Negative deltas mean the diagnostic is faster than full production:

| full L8 batch | splits including new slot | omit merge delta us | serialized merge delta us |
|---|---:|---:|---:|
|16 (Triton control)|33|-6.672|+0.976|
|17|32|-8.928|-2.688|
|24|22|-9.504|-1.792|
|32|17|-8.784|-0.048|
|48|11|-8.240|-1.184|
|64|9|-9.936|-1.552|

B32 complete/omitted **267.984/259.312us**. Omission is strictly an offline
incomplete-output diagnostic, never kernel_fn or a submitted implementation;
its partials are independently reduced in Torch and checked against output/cache
reference before capture, after replay and after timing. Serialization still
computes the full valid output. These measure marginal graph costs, not isolated
merge duration or an exact breakdown of scheduling/traffic. All-key attention
and new cache writes remain in the timed prefix. The serialized control is
nearly flat on scored B32; its nearby B17/B24 savings are evidence for Claude's
owned launch path, not a dispatch change made by Codex.

Native CUDA uses four independent output rows per128-thread CTA, physical
partial-row ordering, coherent volatile FP32 loads after griddepcontrol.wait,
and unchanged normalization/gate/BF16 output. Two concrete layouts:

- **warp:** one row/warp, two dimensions/lane, distributed split max/sum,
  partial vectors loaded before weighted reduction and shuffled weights.
- **striped:** four split groups x eight dimension groups per warp, eight
  dimensions/lane,128-bit vector loads and shuffle reductions across split groups.

Each layout has its own paired production control,60rounds/shape:

| batch | warp parent/native us | warp paired delta us | striped paired delta us |
|---|---:|---:|---:|
|17|161.904/160.432|-1.456|+1.568|
|24|208.336/205.952|-2.288|+0.064|
|32|267.120/266.864|-0.336|+0.528|
|48|400.256/399.088|-1.424|+0.960|
|64|510.016/509.584|-0.304|+0.752|

The scored B32 gain is too small to justify promotion or an experimental check;
striped loses. No follow-up parameter sweep or favorable-seed selection. Median
paired deltas need not equal differences of independently computed medians.

### Validation and limits

- Both wrappers **16/16**, followed by full benches **1.949/1.958x**. All standard
  cases use unchanged production fallback; those suites alone do NOT validate
  the new large-batch merge. Aggregates are experimental local measurements,
  not paired improvements, production regressions or H200 estimates.
- **84 actual native fresh-input graph replays**,42/layout,7cases x6seeds:
  B17/24/32/48/64TP8 and B3/8DP, pages1/64, scales1/1/64, contexts1024–80000,
  soft/coherent/edges, changing QKV/cache/routes/lengths/positions/cache slots,
  including empty contexts. All independent output/cache references pass.
- Isolated native synthetic partial reduction: **48 fresh graph replays**,
  TP8/DP, S5/9/17/22/32/33/64, scales1/1/64, changing partials/maxima/denominators/
  gates, empty partial slots. Independent Torch output comparison passes.
- **Memcheck0 errors, synccheck0 errors, racecheck0 hazards/errors/warnings**,
  each with48 synthetic native replays across both layouts. No Triton loader.
  These isolated checks cover the merge, not attention/cache writes; the
  integrated fresh graph checks above cover the existing PDL launch chain.
- CAP16/32/64 registers warp48/80/32, striped52/62/80; all zero shared memory,
  stack/local allocations and spills. SASS audit verifies all coherent partial
  loads follow ACQBULK in all six specializations. CAP64 has synthetic runtime
  coverage; current Gluon dispatch uses <=32slots. No NCU/dynamic-instruction/
  achieved-occupancy claim; static instruction counts include padding NOPs.
- 12diagnostic entries x80pairs and10native/control entries x60pairs. Custom
  drivers lock generation/reference/capture/replay, flush512MB before each
  timed graph, subtract empty replay. Full benches run sequentially with our
  other GPU work and retain bench.py's timing-loop lock. All local processes
  finished. Generated text whitespace was normalized; audits reran successfully.

### Budget and next cycle

Session **0checks,1scored**, no withdrawals. Codex2026-10-24 totals **4/8checks,
2/3scored**. Final official shared usage **6/10checks,2/10scored**, leaving
4checks/8scored. Actual check cap is10/day. Neither STOP file exists.

1. Resolve **24414e478fa345a2** (current source check) and **78b6db684e514dd6**
   (older exact-source scored request) without resubmitting. Read their passing
   profiles; refresh board and budget before scoring current code. Only one
   Codex scored request remains today. Preserve source identities.
2. Fetch/read main and COORD. Claude owns Triton/Gluon; the nearby serialized-
   merge observations above are available for its review. Current source is
   already validated and unchanged, pending its own H200 check.
3. Close native grouped-row/split-striped large-verify merge replacements.
   There is8–10us marginal headroom, but these layouts recover little at scored
   B32. Do not reopen old small-merge/last-CTA/placement sweeps based solely on
   the long PDL profile rows. No new native architecture is justified by this
   cycle alone; require a distinct measured mechanism before implementing one.
4. Experiment8cd75fa stays separate. Finish on clean pushed codex, shared claim
   removed/result posted; do not wait idle for either queued job.

## Session 23 — 2026-10-24, 02:03–02:10 UTC

### Outcome and production state

Completed a focused native-sliding profiling cycle. **No production change,
no new judge requests.** The remaining source-view shared-wavefront count does
not establish a removable shared-memory bank-conflict bottleneck. A controlled
copy probe reproduces the count by changing global source stride alone, so no
layout candidate or parameter sweep was justified.

Experiment **699f81e** is pushed on **codex-exp/cuda-sliding-conflicts**, starting
at `experiments/sliding_conflicts/README.txt`. Source, nine controlled profiles,
readable raw/source counters, byte checks, isolated diagnostic sanitizers and
reproducible audit are archived. Production evidence copy is
`profiles/session23-native-audit.json`. No experimental code is imported.

Fetched main at start/end; **origin/main3daf4a4** remains already merged.
Production SHA256 remains
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.
Native small-sliding attention and small-full merge, Triton/Gluon and all plans
are unchanged. Existing session21 integrated16/16, bench and fresh validation
therefore apply; no redundant unchanged-production suite or bench was run.

### Counter attribution, not a speedup claim

The native kernel's **4104 source-view excessive shared wavefronts** all belong
to its **four LDGSTS** instructions. Ordinary shared loads/stores and matrix
operations have zero source excess. Reprofiling the exact production native
CUDA reproduces4104/4112 excess/ideal copy wavefronts and zero hardware bank
conflicts for LDGSTS/LDS/STS/GMMA. Zero hardware LDGSTS alone is NOT a proof of
conflict freedom: it also stays zero in the deliberate bad-layout control.

The standalone byte-copy diagnostic varies global source geometry independently
of shared destination mapping. 64CTAs,128threads,4096bytes each, two16-byte
cp.async instructions/thread. Source rows are contiguous64B,64B halves of
successive128B rows, or64B halves of permuted128B rows. Destinations are linear,
production K SW64, or deliberately conflicting128B spacing per16B piece.

| source | linear copy excess | SW64 copy excess | bad-spacing copy excess |
|---|---:|---:|---:|
| contiguous64B rows |0|0|14336|
| 128B row stride |2048|2048|14336|
| permuted128B rows |2048|2048|14336|

Each has2048 ideal copy wavefronts. Bad spacing additionally causes8192 source
LDS excess wavefronts and8192 LDS hardware conflicts. Thus global source geometry
alone changes the derived LDGSTS metric with the destination layout fixed.
The native4104 cannot be read as4104 independently removable shared-bank
conflicts. This corrects a possible interpretation of prior source-counter
totals; those recorded totals themselves remain valid. No precise hardware
implementation, isolated copy latency or end-to-end speedup is inferred.

The probe computes only copies and is never kernel_fn or a submitted candidate.
No invalid/incomplete-output variant was timed against complete attention.
No CUDA graph timing or profiler-duration speedup claim is made this cycle.
Documentation links and metric caveats are in the experiment README.

### Validation and limits

- **27 fresh-data graph replays**,3 per source/destination combination.
  Each independently compares all262144 output bytes against a CPU calculation.
- Permuted-source/SW64 diagnostic only: **memcheck0 errors,synccheck0 errors,
  racecheck0 hazards/errors/warnings**. Each tool executes3 fresh graph replays
  plus one ordinary profile call. This is not new attention sanitizer coverage.
- Every GPU driver holds the shared lock through allocation, generation, capture,
  checks and profiling. Profiling/diagnostic runs are serialized. All processes
  finished; executable and binary NCU reports remain ignored, readable reports
  and reproduction source are committed.
- Audit verifies exact native-source identity, all9 probe checks/counter
  relationships and sanitizer summaries. Generated report whitespace was
  normalized and the audit reran successfully. No native kernel edit occurred.

### Pending judge work, budget and next cycle

Current-source check **24414e478fa345a2** moved to **running** by02:08UTC.
Older exact-source scored job **78b6db684e514dd6** remains **queued**. Neither
was resubmitted; neither yet has a passing profile. Only24414 can qualify current
SHA6b171beb8ed5 for the remaining Codex scored request. Older78b6 is exact45c56bc
SHAe9a877984ef7 and was already legitimately submitted after passing b485.
Do not poll/resubmit resolved b485 or earlier jobs.

Account best remains **2.057970x,rank26**. Session **0checks,0scored**, no
withdrawals. Codex2026-10-24 totals **4/8checks,2/3scored**. Official shared
usage **6/10checks,2/10scored**, leaving4checks/8scored. Actual check cap is10.
Snapshots: `profiles/session23-*`. Neither STOP file exists.

1. Resolve running24414 and queued78b6 without resubmission; read complete
   passing profiles. Refresh board and budget before scoring current source;
   preserve exact source identity and the one remaining Codex scored allowance.
2. Fetch/read main and COORD. No new native architecture is supported by this
   diagnostic alone. Do not reopen shared-layout sweeps based on4104, or the
   closed small-L1 instruction/merge families. Future native topology changes
   need complete graph measurements and a concrete reason to beat production;
   earlier paired-head/full-row native designs remain rejected on their timings.
3. Keep experiment699f81e separate. End on clean pushed codex, remove shared
   claim/post result, and do not wait idle for judge completion.

## Session 24 — 2026-10-24, 02:11–02:19 UTC

### Outcome and production state

Resolved current-source H200 check **24414e478fa345a2** and submitted that exact
source as **fe512312803447b2**. The native new-value staging experiment does not
improve the scored shape, so **production is unchanged**. No new check request.

Fetched main at start/end; **origin/main3daf4a4** remains already merged.
Production SHA256 remains
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.
Native small-sliding attention, its B*Hkv<=16 boundary, small-full reduction,
Triton/Gluon and plans are unchanged. Existing session21 integrated16/16,
bench and fresh validation applies; no redundant unchanged-production suite.

### Exact-source H200 qualification

Check **24414e478fa345a2** completed at **02:13:01UTC**: **16/16,2.082939x**,
CI **[2.051695,2.114183]**, no flags/rejections/errors. Read all12 passing
profiles; complete status/JSON/log and readable profiles are archived under
`profiles/session24-current-status.json` and `profiles/submission.24414*`.
H200 decode1 sliding **5.795us**, decode1/verify1 full **14.895/16.388us**,
verify16/32 full **153.704/259.405us**. Gluon profile: attention249.2us,
merge116.1us,new tokens3.4us,GPU busy260.3us; overlapping rows include waiting
and must not be summed as end-to-end time.

Refreshed board **2.057970x,rank26**, budget6/10checks,2/10scored. Check exceeds
that best, within overlapping confidence intervals. Submitted unchanged
submission.py from this branch as **fe512312803447b2**; its content_hash
`3690964f4e69213c64b1d9903bc486924698d1e57b37fbcae0046f4ea2f1f1b9`
matches check24414 exactly. Existing local validation applies to this same
source. The experimental modules are not imported or submitted.

Older scored **78b6db684e514dd6** (exact45c56bc/SHAe9a877984ef7) and new scored
**fe512312803447b2** remain queued at this writing. Neither is a duplicate
request; do not resubmit them. Check24414 is now resolved and should not be
polled again. No Codex scored allowance remains on2026-10-24.

### Native direct new-value staging: rejected

Experiment **2bdea47** is pushed on **codex-exp/cuda-sliding-direct-newv**, starting at
`experiments/sliding_direct_newv/README.txt`. Sources, reproducible generator,
raw graph pairs, validation, sanitizer reports and SASS/resources are archived.
Production evidence copy: `profiles/session24-native-audit.json`.

The existing native sliding kernel stages new V bytes in raw shared memory,
then synchronizes before other threads convert them into the final FP16 tile.
The **direct** candidate quantizes new V to identical FP8 bytes, expands those
bytes to FP16, and writes the final swizzled tile directly. Other threads skip
that row during conversion. The first barrier still protects zero-filled K
before new K insertion; the final proxy fence/barrier publishes both operands.
This removes the middle barrier without changing precision or any attention.
The **conditional** control retains raw staging but executes the middle barrier
only in the CTA owning the new token (a CTA-uniform predicate).

Balanced adjacent AB/BA cold-L2 CUDA-graph pairs,60rounds/shape,H100 microseconds:

| shape | base/direct us | direct paired delta | conditional paired delta |
|---|---:|---:|---:|
|B1 80K fixed (scored)|4.256/4.288|+0.048|0.000|
|B2 80K agentic|4.960/4.992|0.000|-0.064|
|B3 80K agentic|5.248/5.344|+0.112|+0.096|
|B8 80K agentic|7.264/7.504|+0.192|+0.112|

Each variant has its own paired parent. Short/page64/DP cases are mixed;
no scored benefit warrants promotion, further sweeps or an experimental check.
This does not isolate the barrier's latency from added predicates/instructions.

Page1/EARLY SASS base/direct/conditional has **3/2/3 CTA barrier sites**,
**1056/1088/1056 instructions including padding NOPs**, **52/53/52 F2FP sites**.
All retain2QGMMA/4HGMMA,74registers,17408shared bytes(cuobjdump)/16384(ptxas),
zero stack/local memory and spills. Direct EARLY=False compiles to75registers;
only EARLY=True has runtime coverage. No NCU, dynamic-instruction, occupancy
or isolated barrier-latency claim.

### Validation and handoff

- Both **16/16** (3native,13production fallback), each followed by the full
  broad-native bench: **1.920/1.925x** direct/conditional. Experimental local
  aggregates, not production regressions or H200 estimates.
- **120 all-native fresh graph replays**,60/variant over10cases: changed QKV,
  cache/routes/lengths/positions/cache slots, including empty contexts; TP8/DP,
  pages1/64,scales1/1/64,contexts10–120K,soft/coherent/edge inputs.
- **24 bit-exact partial/cache comparisons**,12case/seed combinations per variant.
- Custom drivers lock generation/reference/capture/replay, flush512MB before
  every timed graph and subtract an empty replay. Full benches run sequentially
  with our other GPU work and retain bench.py's timing-loop lock.
- Source generator and audit verify exact native baseline identity and all
  reported result counts. Final sanitizer/audit completion is recorded below.

Session **0checks,1scored**, no withdrawals. Codex2026-10-24 totals **4/8checks,
3/3scored**. Official shared usage **6/10checks,3/10scored**, leaving4checks/7scored
account-wide; the smaller actual check cap is10/day. CLI remains
`/home/daytona/.okc/bin/okc`; no login, installation or new credentials.

Next cycle: resolve **78b6db684e514dd6** and **fe512312803447b2** without
resubmitting, read their passing profiles and refresh board/budget. No more
Codex scored runs today. Fetch/read main and COORD before source work. Close
this raw-new-V dependency/barrier family along with the earlier L1 micro-work;
it does not justify revisiting shared-layout, scheduling or merge sweeps.
Keep experiments separate, finish on clean pushed codex, remove the shared
claim/post results, and do not wait idle for the judge queue.

Final **02:19UTC**: both variants independently pass **memcheck0errors,
synccheck0errors,racecheck0hazards/errors/warnings**. Each tool checks12native
calls per variant (two pages,two scales,three changed-QKV calls), independently
reducing partials with Torch and checking output/cache. All local processes
finished. Generated report whitespace was normalized; the audit reran and passed.
Experiment2bdea47 is pushed separately. Both78b6 andfe512 remain **queued**;
board2.057970x/rank26,official budget6/10checks,3/10scored. Neither STOP file
exists. Final source hash and already-merged main3daf4a4 are reconfirmed.
Shared claim removed and result posted; finish on clean pushed codex.

## Session 25 — 2026-10-24, 02:20–02:30 UTC

### Outcome and production state

Resolved older scored **78b6db684e514dd6**: **16/16, 2.081734x**, CI
**[2.050508,2.112960]**, no flags/rejections/errors. New account best, **rank22**.
It completed at02:27:28UTC, with content_hash matching its qualifying b485 check.
This is exact older45c56bc/SHAe9a877984ef7, already submitted in session22;
it is not current production and was not resubmitted. Read all12 passing
profiles, saved with complete JSON/log at `profiles/submission.78b6*`.
The CLI originally wrote these beside the frozen session22-qualified source;
they were moved to the canonical profiles directory (text whitespace normalized).

H200 decode1 sliding **5.744us**, decode1/verify1 full **15.333/16.262us**,
verify16/32 full **154.300/263.849us**. Gluon profile: attention249.2us,
merge118.5us,new tokens3.4us,GPU busy260.2us. PDL rows overlap/include waiting;
do not sum them as end-to-end latency. Confidence intervals overlap the old
best; this result does not isolate a native contribution or this cycle's work.

Current-source scored **fe512312803447b2 remains queued**. It is the exact
SHA6b171beb8ed5 source qualified by check24414 (2.082939x). Do not resubmit it.
No more Codex scored requests are allowed today. Older78b6 and check24414 are
resolved; neither needs further polling.

Fetched main at start/end; **origin/main3daf4a4** remains already merged.
Production is unchanged SHA256
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.
No native arithmetic/dispatch, Triton/Gluon implementation, plan or launch changed.

### Native dispatch generalization audit

Experiment **6da6317** (source/results9d68dc4) is pushed separately on
**codex-exp/native-dispatch-audit**, starting at
`experiments/native_dispatch/README.md`. Production evidence copy:
`profiles/session25-native-audit.json`. No experiment is imported by production.
Instead of reopening closed micro-variants, compared complete current production
against frozen current main across **64 distinct nearby shapes**,80 adjacent
AB/BA graph pairs each, plus8 suspect/neighbor/control confirmations at a second
seed with120 pairs each: **6080 total adjacent pairs**.

| H100 shape | main/codex us | paired delta us |
|---|---:|---:|
|sliding L1 B1,80K,page1|4.832/4.416|-0.416|
|sliding L1 B8,80K,page1|7.808/7.536|-0.224|
|sliding L1 B8,80K,page64|7.872/7.264|-0.624|
|full L1 B1,80K,page1|17.808/17.056|-0.720|
|full L8 B1,80K,page1|19.168/18.272|-0.848|
|full L8 B2,80K,page1|38.448/37.712|-0.752|
|full L8 B2,9K,page64|10.016/10.192|+0.192|
|same9K,second seed|10.048/10.304|+0.256|

All24 native-sliding cases favor Codex (-1.376 to-0.192us), including B7/B8,
DP1, pages1/64, scales1/1/64, contexts63–120K, mixed lengths and tile/window
boundaries. The scan includes16 unchanged-fallback controls, with deltas
-0.064 to+0.256us; those are variation, not kernel differences or improvements.

24 native-full cases span -1.920 to+0.192us initially. The small **B2/L8/9K/
page64/agentic/S76 regression repeats**, at about2.5%; within-run block-bootstrap
intervals are [0.112,0.304] then[0.216,0.304]us. Nearby B2/L8/S65/page64 cases
are flat/mixed (-0.160/+0.032 first,-0.000/-0.144 second); page1S65 consistently
wins. B2/L8/40K/page64 still wins by1.584us on confirmation. These points do not
support a broad dispatch cutoff, so none was fitted. **Do not claim production
is never slower than main on nearby shapes.** Record this narrow weakness as a
follow-up lead, without reopening merge architecture or main's plans.

Exact B1 S64/S65 transitions pass and show native gains: L1 ctx8191/8192
page1/page64 (-0.480/-0.256us), L8 ctx8184/8185 page1 (-0.720us).
The8 new tokens count toward capacity: L8ctx8191/8192 probes are bothS65,
not an exact transition. Paired deltas are not differences of separate medians.
Bootstrap intervals describe within-run variation, not all system variation,
multiple-comparison correction or H200 confidence. No aggregate speedup claim.

### Validation and driver correction

- **432 independent output/cache reference checks**: both complete kernels
  before capture, after first replay and after timing on all72 entries.
- **192 corrected fresh graph replays**,24 shapes x4 inputs x2 implementations.
  Codex96 comprise28native sliding,40native full and28fallback; main96 are
  separate controls. Changed QKV/cache/routes/lengths/positions/cache locations,
  shortened and empty contexts, TP8/DP, pages1/64, scales1/1/64 all pass.
- Initial fresh driver aborted with an illegal memory access after one shape.
  Found a validity defect: randint(0,ctx+1) could extend an agentic request beyond
  its generated length into unassigned page-table entries. Corrected it to
  shorten each generated request, assert nonnegative new rows and log replay
  context. The original failing random draw was not recorded, so the particular
  fault's cause is not conclusively attributed. Failed log/partial result stay
  archived as fresh-invalid-lengths.*, excluded from pass counts. Timed inputs
  never used that mutation and are unaffected. Both sources pass the corrected
  full fresh suite; no kernel change was made in response to the driver fault.
- GPU lock covers generation/reference/capture/replay;512MB flush precedes
  every timed graph, empty replay subtracted. No concurrent own GPU work.
- CPU audit verifies exact production hash, all14 shared top-level helper ASTs,
  complete result/pair/replay counts. No new sanitizer or redundant unchanged
  production16-case suite; prior native sanitizers/current-source validation apply.
  All local processes completed. Source and reproducible raw reports are pushed.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 totals remain
**4/8checks,3/3scored**. Official shared usage **6/10checks,3/10scored**,
leaving4checks/7scored account-wide. Actual check cap10/day. Neither STOP file.

1. Resolve **fe512312803447b2** without resubmission and read its passing profile.
   Only this scored job remains pending. Do not poll resolved78b6/24414/b485.
   No more Codex scored requests today, regardless of remaining account budget.
2. Fetch/read main and COORD. Production source is unchanged, already checked
   and submitted; preserve source identities and respect Claude's ownership.
3. If pursuing the small nearby full-merge weakness, first investigate a broad
   neighborhood of B2/L8/S~76 and page1/64 with complete graph controls. A single
   context or page-specific cutoff is not justified. This has no scored impact;
   avoid consuming checks solely for an unchanged scored path. The sliding
   boundary remains supported; closed L1 micro-optimization families stay closed.
4. Keep experiment6da6317 separate. Finish on clean pushed codex, shared claim
   removed/results posted. Do not wait idle for the remaining queued job.

## Session 26 — 2026-10-24, 02:32–02:44 UTC

### Outcome and production state

Resolved older scored **fe512312803447b2: 16/16,2.091142x**, CI
**[2.059775,2.122509]**, no flags/rejections/errors; new account best, **rank20**.
Read all12 passing profiles and saved complete JSON/log/text as
`profiles/submission.fe512312803447b2.*`. Its content_hash matches qualifying
check24414 exactly. This is older SHA6b171beb8ed5, not either source integrated
this cycle, and was not resubmitted. Confidence intervals overlap the previous
best; do not attribute this independent aggregate increase to today's work.

H200 decode1 sliding5.91us, decode1/verify1 full14.32/16.31us, verify16/32
full156.15/260.01us. Final Gluon profile: attention251.7us, merge120.3us,
new tokens3.4us, GPU busy262.8us. PDL rows overlap/include waiting and cannot
be summed as graph latency.

Integrated two new main improvements; **no native implementation or dispatch
change**. Initial main **a794ac6** merged as **8c35564**: Gluon full verify from
six pairs up, one wave unless two fill slots materially better. A late fetch
then found **824e52a**, merged as **d8bab0a**: mask only each chunk's last tile
in the Gluon consumer. Both are Claude's changes; Codex did not tune its lane.

Final production SHA256:
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Final integrated suite **16/16**, full local bench **1.987x** against fixed
SGLang/H200 reference times, verify16/32 full **139.50/260.08us**. The unchanged
native small-sliding path and small-full reduction retain prior sanitizer
coverage. No new sanitizer claim. No experimental import was added.

Experiment **7305b35** is pushed on **codex-exp/native-merge-neighborhood**;
initial source/results **af33923**. Start with
`experiments/native_neighborhood/README.txt`. Frozen previous, initial merged,
final merged and main snapshots preserve all source identities. Drivers,
raw pairs, validation logs and CPU audits are archived separately from production.
Production evidence copies: `profiles/session26-{native,final}-audit.json`.

### Main integration evidence

Initial merged SHA
`6f0fcb357146bbfd4e69e3f9d8a3ea2d9968f09c26b5876a0aa83af63ebe8c05`
passed **16/16**, full bench **1.965x**, and **104 fresh graph replays** across
13 full-verify shapes, four changed inputs per implementation. Current-source
52 calls comprise44 Gluon and8 Triton fallback; previous-source52 are separate
controls. All QKV/cache/routes/lengths/positions/cache slots change, with shortened
and empty contexts. Pages1/64, TP8/DP and mixed lengths are included.

Balanced adjacent AB/BA cold-L2 graph pairs,80 rounds/shape,H100 microseconds:

| full L8 batch,80K | previous / initial merged us | paired delta us |
|---|---:|---:|
|6|68.432/69.840|+1.312|
|8|85.200/83.280|-2.176|
|12|122.384/115.968|-6.512|
|16|153.264/145.296|-7.824|
|17|160.720/155.664|-5.040|
|24|211.856/204.928|-7.216|
|32|273.408/268.208|-4.912|
|64|515.312/512.016|-3.792|

B4/B5 and B40 are unchanged controls. DP2/20K/page64 improves2.736us; B6/4K/
mixed/page64 is nearly flat (+0.112us). Three second-seed confirmations,120pairs
apiece: B5 control+0.080us, B6 **-1.056us**, B8-4.240us. Thus B6's initial
regression reverses with fresh inputs; retain both observations for Claude,
without claiming a universal B6 gain/regression or changing its plan.

The late824e52a merge independently passed16/16 and adds **56 fresh graph
replays**,28/source over7shapes (all final28 use Gluon). Cases include
B6/8/16/17/32 at80K,DP2/page64/scale1/64 at20K,and B6 mixed4K/page64 with edge
keys. Changes include shortened and empty contexts. Seven additional paired
comparisons,80rounds/shape, isolate this second main update:

| full L8 batch | initial merged / final us | paired delta us |
|---|---:|---:|
|16|145.584/139.920|-5.808|
|17|154.880/147.024|-7.760|
|32|268.016/257.584|-10.320|

Do not add deltas from separate paired runs to estimate a combined speedup.
These are local graph measurements, not H200 estimates. Final audit verifies
all20 shared helper ASTs, including nested Gluon functions, against main824e52a,
and exact unchanged native extension/dispatcher ASTs. No Codex kernel edit.

### Native full-merge neighborhood: no cutoff promoted

Followed session25's small B2/L8/page64 weakness across **44 shapes**: B1/fixed
and B2/agentic at7K,8192,9K,10K,12K,16K,20K,40K on pages1/64, plus B2/fixed at
8192,9K,10K,12K,16K,20K on both pages. Complete current-source kernels compared
with frozen main; the only exercised implementation difference is native merge.
Initial80pairs/shape, then nine suspect/neighbor/control confirmations at a second
seed with120pairs each.

All20 native page1 cases favor native (-1.136 to-0.272us); page1 fallback controls
are flat. All7 native B1/page64 cases win (-0.896 to-0.576us). B2/page64 remains
flat or slightly slower at some small split counts:

| B2/page64 case | actual splits | initial / confirmation paired delta us |
|---|---:|---:|
|8192 agentic|69|+0.032/+0.128|
|9K agentic|76|0.000/+0.144|
|10K agentic|85|-1.248/-1.632|
|8192 fixed|65|0.000/+0.032|
|9K fixed|71|-0.320/-0.272|
|10K fixed|79|-0.960/-0.960|
|16K agentic|68|+0.320/no second-seed check|

Actual splits are not monotonic in context: chunk-size transitions reduce them.
Page1 at the same agentic contexts/split counts wins on both seeds. B2/page64
40K still wins1.872us initially. This supports a narrow small-split weakness,
not a shape-general cutoff; **no native dispatch change**. Do not claim Codex
is always faster than main on nearby shapes. Closed native micro-optimization
and merge architecture families remain closed.

Initial integration+neighborhood:57 distinct shapes,12confirmation entries,
**6000 adjacent pairs,414 independent output/cache reference checks**. Final
integration adds560pairs/42checks. Total fresh replays160 across both phases,
including separate baseline controls. Each custom driver locks generation,
reference,capture,replay,flushes512MB before every timed replay,and subtracts an
empty replay. Full benches ran sequentially with own GPU work, retaining
bench.py's timing-loop lock. All local GPU processes completed. Bootstrap
intervals quantify within-run variation only, not H200 uncertainty or multiple
comparisons. Audits verify exact hashes and complete log/result counts.

### Judge jobs, budget and next cycle

Requested **one check3e786669e1bb41a8** at02:36UTC for exact **8c35564/SHA6f0f**,
after initial local validation and paired wins. Its content_hash is
`b20f39f4b465efaed561959a77cf7a502b68f6f3174a76c4951ed3bd80b74047`.
It remains **queued**. **It does NOT qualify final d8bab0a/SHA1bcf42aa6edf**:
824e52a arrived after the request. Preserve exact snapshots if acting on results.
No second check was requested for the late merge; official shared usage rose to
8/10checks,leaving2. An additional account check2363a04971ab46b1 appeared after
ours; it was not a Codex request. Do not assume it qualifies Codex's native source.

Older scored **fe512312803447b2 resolved at the final02:43UTC read**, as
described above. Only check3e786 remains pending. Account best is now
**2.091142x,rank20**, replacing resolved78b6's2.081734x. No more Codex scored
requests today: session **1check,0scored**, Codex2026-10-24 totals **5/8checks,
3/3scored**. Official account **8/10checks,3/10scored**,leaving2checks/7scored.
The actual check cap is10/day. No withdrawals or STOP files.

1. Resolve queued3e786 without resubmitting; read its passing profile.
   Resolvedfe512/78b6/24414/b485 and earlier jobs need no further polling.
2. Final production SHA1bcf42aa6edf is locally validated but has NO exact-source
   qualifying check. Older check3e786 covers only frozen8c35564/SHA6f0f. No scored
   allowance remains today regardless of account budget. Preserve the remaining
   shared checks until a stable source merits H200 confirmation.
3. Read COORD/fetch main before work. Native small-full weakness is small and
   page/row-count dependent; do not fit a context-specific cutoff. No new native
   architecture is justified by this audit alone. Reported B6 Gluon variation
   belongs to Claude's lane.
4. Keep experiment7305b35 separate; finish clean on pushed codex, remove shared
   claim and post results. Do not wait idle for the remaining queued check.

## Session 27 — 2026-10-24, 02:45–02:56 UTC

### Outcome and production state

Diagnosed the small-full native merge's page-dependent weakness. **No production
change, dispatch cutoff, or new judge request.** Serializing the native merge
fails to repair the weakness and slows both scored small-full shapes. Keep PDL.

Experiment **a511b1f** is pushed separately on **codex-exp/native-merge-pdl**;
start with `experiments/native_merge_pdl/README.md`. Complete frozen sources,
raw graph pairs, fresh validation, Nsight Systems intervals, CUDA occupancy
queries, SASS and a reproducible CPU audit are archived. Production evidence:
`profiles/session27-native-audit.json`. No experimental import was introduced.

Fetched main repeatedly; **origin/main824e52a** remains already merged. Final
production SHA256 is unchanged:
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
No native implementation/dispatch or Triton/Gluon source/plan changed. Existing
session26 integrated16/16 and bench validation therefore still applies; no
redundant unchanged production suite/bench or new sanitizer claim this cycle.

### Complete-graph launch controls

Four complete functions: main/native with existing PDL, plus serialized-merge
controls for each. Native serialization selects its existing `pdl=False` kernel
specialization; main serialization overrides only merge PDL/launch flags in an
offline wrapper. Attention, every key, precision, cache writes and output remain
inside timed calls. The main control is diagnostic, not a Triton plan change.

20 distinct shapes,80 adjacent AB/BA pairs per contrast, four contrasts per shape;
eight second-seed confirmations,120 pairs each. **10240 total adjacent pairs**.
Shapes span pages1/64, B1 decode/verify80K, B2 verify agentic8192/9K/10K/16K/40K/
80K, B2 fixed9K and B1 fixed9K. All use the actual native full-merge dispatch.

H100 paired deltas in microseconds, initial / confirmation:

| case | native minus main, PDL | serial native minus native PDL |
|---|---:|---:|
|B1 L1 80K page1 (scored)|-0.048/-0.064|+0.800/+0.608|
|B1 L8 80K page1 (scored)|-0.464/-0.496|+0.656/+0.592|
|B2 L8 8192 agentic page1|-0.912/-1.152|+0.416/+0.736|
|B2 L8 9K agentic page1|-1.184/-1.088|+0.800/+0.672|
|B2 L8 16K agentic page1|-0.880/-0.320|+0.192/+0.288|
|B2 L8 8192 agentic page64|+0.096/-0.128|+0.512/+0.480|
|B2 L8 9K agentic page64|+0.192/+0.128|+0.032/+0.144|
|B2 L8 16K agentic page64|+0.416/+0.208|+0.224/+0.480|

The9K/16K page64 weakness repeats, while8192 changes sign. Serialized main versus
serialized native at9K differs only-0.096/-0.160us on page64 and+0.032/-0.064us on
page1. Launch-chain behavior contributes to the difference; serializing native
is not a useful remedy. PDL toggles change both launch behavior and compiled wait
specialization, so this is not an isolated wait-cost or register-contention proof.
No threshold was fitted. Paired deltas are not differences of separate medians;
bootstrap intervals describe within-run variation, not H200 uncertainty. No
aggregate speedup claim, and no claim Codex is always faster than main nearby.

### Scheduling mechanism and validation

For **B2/L8/9K/S76**, CUDA's `cuOccupancyMaxActiveBlocksPerMultiprocessor` gives:

| attention page | registers/thread | shared bytes | max active CTAs/SM | grid/132SM slots |
|---|---:|---:|---:|---:|
|1|170|62464|2|304/264|
|64|168|62464|3|304/396|

These are theoretical residency limits, not measured achieved occupancy. Both
sources and both launch controls have identical attention SASS within each page,
verified from saved full disassembly, despite different compilation-cache hashes.
Native merge uses256threads/48registers, main128threads/56registers; both launch
256 output CTAs in this case. No attention kernel was edited to get these counts.

Nsight Systems node tracing, three repetitions of each complete graph, shows
main/native merge starting **6.368/6.272us** after attention starts on page1,
but **0.864/0.960us** on page64. PDL overlap is3.840/3.648us versus7.520/8.032us.
Serialized controls have zero overlap and merges around3.2us. These instrumented
kernel durations are **not benchmark timings**, and rows include waits; do not
sum them as graph latency. The grid-capacity difference supports the observed
earlier page64 overlap, without proving an isolated hardware bottleneck or a
universal dispatch rule. This explains why page and split count alone were poor
cutoff candidates in the preceding two audits.

- **336 independent output/cache checks** in the timing drivers (four modes,
  three checkpoints,20 initial+8 confirmation entries).
- **320 fresh graph replays**,20 cases x4 changed inputs x4 implementations,
  including changed QKV/cache/routes/lengths/positions/cache slots and shortened/
  empty contexts. Each is independently checked against output/cache reference.
- **24 instrumented complete graph replays**,16 independent timeline checks;
  final occupancy driver8 independent checks. Initial occupancy collection was
  rerun while adding SASS evidence; these counts refer to final retained drivers.
- Initial SASS hash parser expected cuobjdump address comments, whereas Triton
  emits decoded lines; it hashed empty strings. Those fields are invalid in
  archived `occupancy-invalid-hash-parser.*`. Actual occupancy/output observations
  were unaffected. Final parser hashes full saved SASS after whitespace cleanup;
  CPU audit verifies actual bytes/equality. No timing conclusion used bad hashes.
- GPU lock covers generation/reference/capture/replay;512MB flush precedes every
  timed graph, empty replay subtracted. Profiling/occupancy run after benchmarks
  and fresh tests finish. All local processes completed. Audit verifies frozen
  source identities, all20 shared helper ASTs, result counts and residency/SASS.

### Judge jobs, budget and next cycle

Existing **check3e786669e1bb41a8** moved from queued to **running**, starting
**02:51:37UTC**. No result or passing profile yet; do not resubmit. It covers only
**8c35564/SHA6f0f**, before main824e52a, and **cannot qualify current SHA1bcf42aa6edf**.
Resolvedfe512/78b6/24414/b485 and earlier jobs need no further polling.

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 totals remain
**5/8checks,3/3scored**. Official account **8/10checks,3/10scored**, leaving
2checks/7scored account-wide. Actual check cap10/day. No more Codex scored runs
today regardless of account balance. Board best **2.091142x,rank20**. No STOP files.

1. Resolve running3e786 and read its passing profile without resubmission; preserve
   exact-source identities. Current source has no exact qualifying H200 check.
2. Read COORD/fetch main before work. Preserve remaining shared checks while main
   is evolving; Claude is still working on Gluon verify-full. No other pending
   Codex jobs exist.
3. Close simple native small-full serialization as a remedy. The residency/PDL
   observation is useful diagnostic evidence, not authorization to reopen the
   closed reduction, tile, load-placement or native-sliding micro-sweeps. Any new
   implementation needs a distinct mechanism and broad complete-graph evidence.
4. Keep experimenta511b1f separate, finish clean on pushed codex, remove claim and
   post results. Do not wait idle for the running judge job.

## Session 28 — 2026-10-24, 02:56–03:12 UTC

### Outcome and production state

Resolved the remaining Codex judge check and rejected a distinct native
**full-attention L1 single-large-key-tile** design. **No production kernel or
dispatch change; no new judge requests.** Experiment **081f126** is pushed on
**codex-exp/cuda-full-decode-tile**. Start with
`experiments/full_decode_tile/README.md`; source generation, frozen sources,
raw graph pairs, reference/fresh/sanitizer logs, SASS and CPU audit are there.
Production evidence copy: `profiles/session28-native-audit.json`.

Fetched main at start and end. Initial **824e52a** was already merged; late
**ab141cd** adds only Claude's session11 notes and was merged into codex.
Read those notes: no further kernel update. Production remains SHA256
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Native small-sliding and small-full kernels, all Triton/Gluon helpers/plans and
production dispatch are unchanged. Existing session26 integrated16/16 and bench
validation applies; no redundant unchanged-production run or new production
sanitizer claim. No experimental imports were introduced.

### Resolved H200 check

**3e786669e1bb41a8: 16/16,2.107493x**, CI **[2.075881,2.139105]**,
no flags/rejections/errors. Read all12 passing profiles; full artifacts saved as
`profiles/submission.3e786669e1bb41a8.*`, status snapshots under session28.
This is **8c35564/SHA6f0fcb357146**, maina794ac6 plus native, content_hash
`b20f39f4b465efaed561959a77cf7a502b68f6f3174a76c4951ed3bd80b74047`.
It predates main824e52a's last-tile mask and **cannot qualify current SHA1bcf**.
No resubmission: Codex's scored allowance is exhausted today. Its interval
also overlaps the account best; do not claim a statistically established rank
improvement or attribute independent aggregate differences to native code.

H200 decode1 sliding/full **5.585/15.198us**; verify1 full **16.327us**;
verify16/32 full **141.642/264.970us**. Final Gluon profile: attention259.0us,
merge225.5us,newtokens3.2us,GPUbusy267.5us. PDL rows overlap and include waits;
do not add them as graph latency. **No pending Codex jobs remain.** Claude's
separate check2363a04971ab46b1 is its job/source, not an exact Codex check.

### Native full-decode experiment

The earlier native attention pipelines targeted L8. This cycle instead tests
L1 full attention as independent128/256-key tiles: each128-thread CTA issues
all gathers before query normalization, uses FP8 m64n128/m64n256 QK and FP16
PV, and produces FP32 partials without a streaming loop or running softmax.
The new token is inserted/written in its natural tile. All keys and values are
included, FP8 query/cache precision and BF16 output preserved. The unchanged
inherited merge uses PDL; no Triton/Gluon plan or source was tuned. Both wrappers
are intentionally broad L1/full experiments, not proposed production dispatches.

60 adjacent balanced AB/BA cold-L2 graph pairs per shape/width, H100 us:

| full L1 | parent/tile128 | paired delta | parent/tile256 | paired delta |
|---|---:|---:|---:|---:|
|B1 80K page1|16.976/22.944|+6.000|17.040/25.584|+8.560|
|B2 80K agentic page1|28.336/41.968|+13.440|28.432/44.064|+15.600|
|B16 80K agentic page1|129.920/229.552|+99.600|129.760/256.624|+126.752|
|B32 80K agentic page1|240.928/448.016|+206.816|240.848/501.904|+261.248|
|B1 80K page64|16.720/22.672|+6.032|16.640/23.936|+7.200|
|B2 9K page64,scale1/64|11.088/8.544|-2.608|11.184/10.048|-1.136|
|B1 ctx10 page1|5.072/4.448|-0.608|5.072/5.824|+0.704|
|DP1 ctx4096 page64,scale1/64|11.584/12.528|+1.056|11.504/14.848|+3.360|

Each width has its own paired parent. Deltas need not equal differences of
separate medians. Scored losses are decisive; no favorable-seed repeat or
parameter sweep. Incidental short-context gains do not establish a general
dispatch rule. Broad-wrapper full benches **1.619/1.541x** versus fixed
SGLang/H200 times are experimental results, not production regressions.

Page1/EARLY uses112/224registers,32768/65536dynamic shared bytes plus1024static
bytes reported by cuobjdump; zero stack/local allocation and spills. SASS has
two QGMMA each and8/16HGMMA,LDGSTS,vector V stores and raw-V loads. No NCU,
achieved-occupancy or isolated cost claim. Larger tiles remove streaming work
but expand matrix/softmax work, register pressure and the total number of
partials versus production. This topology does not win on scored full decode.

### Validation and a generator bug

Both corrected wrappers **16/16**, each5actual-native/11fallback cases; each
followed by the full12-shape bench. **72 independent output/cache checks** in
8shape/3mode comparisons (before capture,after first replay,after timing),
**960 adjacent graph pairs**. **48 actual-native fresh graph replays**,6shapes
x4inputs x2widths, with changed QKV/cache/routes/lengths/positions/cache slots,
shortened and empty contexts, DP,pages1/64,scales1/1/64.

Final isolated native **memcheck0errors,synccheck0errors,racecheck0hazards/errors/
warnings**;24fresh graph calls per tool, both widths/pages/scales, including
empty contexts. Torch independently reduces FP32 partials and checks output/
cache reference; no Triton loader kernels. Eight additional final128-width
boundary output/cache checks and independent partial comparisons pass at
contexts10/63/64/127/128/255/256/1024.

Initial128-key generated source FAILED correctness before timing. Broad
sequential replacements changed gather2->4, then accidentally4->8 while
widening the conversion loop:256rows copied into128-row buffers, overwriting
adjacent shared arrays. QK scores/maxima matched while V/PV failed. Pinning
probability registers and synchronous PV controls did not fix it. The final
generator scopes the conversion edit; audit checks exact gather byte capacity
and disjoint shared buffers. Final source uses dynamic shared memory for both
widths, enabling the256-key variant's64KB allocation.

Pre-fix sources/debug logs are explicitly invalid evidence in the README.
The early memcheck reported zero address errors while numerical checks failed;
it did not catch logical-array overlap inside shared allocation, and is NOT
final correctness coverage. No failed source was timed/submitted, no withdrawal.

Custom GPU drivers lock generation/reference/capture/replay;512MB flush before
each timed graph and empty replay subtraction. Standard suite/bench drivers
retain bench.py's timing-loop lock, with no concurrent own GPU work. All local
processes completed. CPU audit reproduces generated sources, verifies frozen
production identity and14 shared top-level helpers against main, checks counts
and sanitizer logs. This is not an all-nested-helper AST audit. SASS and raw
reports are archived; no further production validation claim is implied.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 totals remain
**5/8checks,3/3scored**. Official account **8/10checks,3/10scored**, leaving
2checks/7scored account-wide; actual check cap10/day. Account best remains
**2.091142x,rank20**. Neither STOP file exists. CLI is the existing
`/home/daytona/.okc/bin/okc`; venv activation alone does not add it to PATH.

1. All Codex judge jobs are resolved. Do not poll/resubmit3e786 or older jobs.
   CurrentSHA1bcf still has no exact-source H200 check. Preserve remaining shared
   checks while no new scored allowance remains; Claude2363 is a different source.
2. Read COORD/fetch main. Close large single-tile native full-decode as a scored
   optimization. Existing closed sliding instruction/reduction/PDL families
   remain closed; future work needs a distinct measured mechanism.
3. The short-context B2 gain is only one scaled/page64 observation plus B1ctx10;
   no short-context cutoff is justified. Any investigation must first cover
   broad contexts, pages, scales, batches and complete graph controls. It has no
   demonstrated scored benefit and should not alone consume scarce checks.
4. Keep experiment081f126 separate. Finish clean on pushed codex, remove the
   shared claim and post results. No pending Codex job requires waiting.

## Session 29 — 2026-10-24, 03:12–03:20 UTC

### Outcome and production state

Audited session28's short-context full-L1 tile128 gains and tested composition
with the existing native full merge. **No production kernel/dispatch change,
no judge request.** Experiment **a93b1ed** is pushed separately on
**codex-exp/full-decode-neighborhood**; start with
`experiments/full_decode_neighborhood/README.md`. Frozen sources, complete raw
graph pairs, fresh checks, suite/bench logs, and CPU audit are archived there.
Production audit copy: `profiles/session29-native-audit.json`.

Fetched main at start and end; **origin/main ab141cd** is already merged.
Production remains SHA256
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
No native CUDA kernel, Triton/Gluon helper or plan changed. Current production's
session26 validation remains applicable; no redundant production run or new
sanitizer claim. No experimental imports were introduced.

### Broad short-context audit

Unchanged session28 tile128 wrapper versus production across **74 shapes**:
B1/2/4/8, contexts128/1024/4096/9K/16K/32768, pages1/64; matched9K
scale/distribution controls; DP, tile boundaries and80K controls. Initial60
adjacent AB/BA pairs each;22 representative cases confirmed with100pairs each
at a second seed. H100 paired deltas, initial/confirmation, microseconds:

| batch/context/page | tile128 minus production |
|---|---:|
|1/4096/1|-0.480/-0.352|
|1/9K/1|+0.208/+0.224|
|1/9K/64|0.000/+0.160|
|2/4096/1|-1.824/-2.048|
|2/9K/1|-2.000/-2.240|
|2/9K/64|-2.864/-2.912|
|2/16K/1|+0.048/-0.672|
|4/9K/1|+0.352/+0.016|
|8/4096/1|+0.816/+0.192|

B2/9K wins across both scales(1,1/64), pages, soft/coherent distributions:
initial-3.056 to-2.000us, confirmation-3.040 to-2.192us. Thus session28's gain
was not confined to one scaled/page64 input. It does not generalize uniformly
across batch/context. Both DP/4096 controls lose(+1.248/+0.848us).
B1/80K scored control still loses6.448us; B2/80K/page64 loses10.448us.
No context cutoff or production extension was fitted/promoted.

### Existing-native-merge composition

Production's existing native full merge applies when B*Hkv<=4 and64<S<=128,
but the frozen tile wrapper always used Triton merge. Applied the **same
predicate** to the tile wrapper's actual S without changing either kernel.
Sixteen B1/2 cases at4096/8192/9K/16K, both pages, compare the composition with
both production and original tile.80pairs/contrast; a second seed reverses
graph capture order and repeats all contrasts.10cases activate native merge,
6are identical-code controls. Wrapper AST audit verifies this sole change
besides its default tile width.

Composition minus original tile: active cases save0.512–0.656us initially and
0.416–0.768us on confirmation. B1/9K now beats production:
page1-0.272/-0.464us, page64-0.208/-0.528us. This explains part of the earlier
batch-dependent boundary; it is not an isolated hardware latency attribution.
Identical-code controls initially had+0.016 to+0.160us offsets, reversing to
-0.064 to+0.016us under new seed/reversed capture order. Fine timing differences
remain sensitive to allocation/capture effects. No favorable-repeat selection.

The composition leaves the losing scored80K topology unchanged. Its broad
wrapper bench1.622x against fixed SGLang/H200 numbers is an experimental result,
not a production regression/H200 estimate. Preserve it as a useful short-context
implementation building block, not a scored candidate or a justified cutoff.

### Validation, budget and next cycle

**11,760 adjacent graph pairs**, **864 independent output/cache checks** at
comparison checkpoints. Original tile64 fresh replays plus composition64:
changed QKV/cache/routes/lengths/positions/cache slots, shortened/empty contexts.
40composed replays use native merge;24retain Triton merge. Composed full suite
**16/16**,5native-attention cases,1also using native merge; followed by all12
bench shapes. No CUDA kernel changed and no sanitizer run was added.

Custom GPU drivers lock generation/reference/capture/timing, flush512MB before
every timed replay and subtract empty replay cost. Suite/bench retains bench.py's
timing-loop lock, with no overlapping own GPU work. All processes completed.
Audit verifies exact inherited sources against081f126, unchanged production,
wrapper AST difference, all counts and paired medians. Bootstrap intervals in
the archive describe within-run variation only, not H200 or multiple comparisons.

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 remains
**5/8checks,3/3scored**. Final official account **8/10checks,4/10scored**, leaving
2checks/6scored account-wide; actual check cap10/day. Account best still
**2.091142x,rank20**. COORD reports Claude check2363 resolved16/16,2.130x and
its exact-source scored request **d0f30dc0d435455d** pending. These are Claude's
source/jobs, not an exact Codex qualification; no duplicate request or polling.
No Codex jobs remain pending. Neither STOP file exists.

1. Read COORD/fetch main; leave Claude's pending score to its owner. Current
   production SHA1bcf still lacks an exact-source H200 check; preserve scarce
   checks and do not exceed exhausted Codex scored allowance today.
2. Close this short-context audit as a scored direction. No repeat tile-width,
   stage, merge-microarchitecture or fitted-cutoff sweep is justified. The
   existing-merge composition may help future generality work, but provides no
   demonstrated scored benefit.
3. Keep experimenta93b1ed separate. Finish on clean pushed codex; remove the
   shared claim and post results. No pending Codex job requires waiting.

## Session 30 — 2026-10-24, 03:20–03:27 UTC

Completed a focused **native sliding CTA traversal-order** cycle. **No production
change or judge request.** Experiment **ee958a2** is pushed separately on
**codex-exp/native-cta-order**. Start with `experiments/cta_order/README.md` on
that branch; frozen sources, raw graph pairs, validation logs, disassembly,
resource reports and NCU counters are archived. Production audit copy:
`profiles/session30-native-audit.json`.

Fetched main at start and end; **origin/main ab141cd** is already merged.
Production SHA256 remains
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Native implementations/dispatch, Triton/Gluon and plans remain unchanged.
Session26 production validation still applies; no redundant production-only
suite or new sanitizer claim. No experimental import was introduced.

### Experiment and result

Changed native grid `(S,B,Hkv)` to `(Hkv,S,B)`, with matching block-coordinate
reads, to launch the two KV heads of each split next to each other. The hypothesis
was improved cache-row locality; actual hardware scheduling is not guaranteed.
The generator makes exactly those two source changes plus the extension name.
Attention, all keys, precision, cache writes, partial layout and PDL are identical.
This is not a Triton plan change or a repeat of in-kernel load-order experiments.

Sixteen shapes, two complete-graph contrasts each (candidate and identical-code
capture control), 80 adjacent AB/BA pairs. Eight representative cases repeated
at a second seed with reversed capture order, 120 pairs/contrast. H100 paired
deltas in microseconds, candidate minus production, initial/confirmation:

| Shape | Candidate | Identical-code control |
|---|---:|---:|
| B1,80K,page1 (scored) | 0.000/-0.096 | +0.080/-0.032 |
| B2,80K,page1 | -0.032/-0.128 | +0.032/-0.080 |
| B8,80K,page1 | -0.112/-0.096 | +0.032/0.000 |
| B1,80K,page64 | +0.048/0.000 | 0.000/-0.192 |
| B8,80K,page64 | -0.192/-0.288 | +0.048/-0.128 |
| DP1,4096,page1 | +0.032/-0.128 | +0.160/-0.096 |
| DP1,4096,page64,scale1/64 | 0.000/-0.192 | -0.016/-0.096 |

Small offsets between identical implementations limit interpretation. B8 has a
modest nearby gain, but no clear scored improvement warrants promotion/checks.
No cutoff or additional traversal sweep. Paired deltas are not differences of
independent medians. Candidate full bench **1.980x** against fixed SGLang/H200
numbers is unpaired experimental evidence, not a production regression or H200
speedup estimate.

### Mechanism and validation

Both sources use74registers and17408shared bytes (cuobjdump), zero local/stack
allocation. Among4232 decoded SASS sites across all four page/EARLY variants,
only12 coordinate reads change; all other instruction text is identical.
CPU audit verifies this and all21 Python helper/function ASTs, including nested
Gluon functions. EARLY=True is runtime tested; EARLY=False is compiled only.

Six isolated NCU launches, after all timing/fresh work, with independent output/
cache checks. Production/candidate HBM read bytes: B1page1 **553216/553472**,
B8page1 **4300032/4299776**, B8page64 **4236288/4236544**. Executed instructions
are identical within each pair (164482/1315856/1336336 by shape); L2 read sectors
have no consistent reduction. This does not support an HBM read-amplification
bottleneck. No achieved-occupancy or isolated cache-latency attribution, and no
profiler duration is treated as graph timing.

- Candidate **16/16**, three actual-native/thirteen fallback cases, followed by
  all12 benchmark shapes.
- **4480 adjacent graph pairs**, **216 independent output/cache reference checks**
  at timing checkpoints, plus six profiler output/cache checks.
- **64 actual-native fresh graph replays**,16shapes x4inputs, TP8/DP,pages1/64,
  scales1/1/64,soft/coherent/edges,contexts10–120K. QKV/cache/routes/lengths/
  positions/cache slots change; shortened and empty contexts included.
- Custom drivers lock generation/reference/capture/replay, flush512MB before
  every timed graph and subtract empty replay cost. Suite/bench uses bench.py's
  timing-loop lock, with no concurrent own GPU work. All processes completed.
- No new sanitizer run: this is a bijective CTA remapping, with unchanged address
  formulas, allocations and synchronization. Frozen-source/count/SASS audit passes.
  Only whitespace/EOF normalization was applied to generated text reports.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 stays **5/8checks,
3/3scored**. Official account **8/10checks,4/10scored**, leaving2checks/6scored;
actual check cap10/day. Board best remains **2.091142x,rank20**. Claude's scored
**d0f30dc0d435455d** belongs to its owner; no duplicate polling/request. No pending
Codex jobs. Neither STOP file exists.

1. Read COORD/fetch main next cycle. Current SHA1bcf still has no exact-source
   H200 check. Preserve shared checks and respect exhausted Codex scored budget.
2. Close head-adjacent native traversal as a scored direction: no traffic win,
   only modest nearby gains. Do not extend into grid-order or cutoff micro-sweeps.
   Earlier closed attention/reduction/load-placement families stay closed.
3. Keep ee958a2 separate, finish on clean pushed codex, remove shared claim and
   post results. No Codex judge job requires waiting.

## Session 31 — 2026-10-24, 03:28–03:36 UTC

Completed a focused **single-tile native sliding cluster/DSM fusion** experiment.
**No production kernel or dispatch change; no judge requests.** Experiment
**22ca912** is pushed on **codex-exp/native-cluster-tile**. Start with
`experiments/cluster_tile/README.md` on that branch; generators, frozen sources,
raw graph pairs, fresh checks, sanitizer logs, SASS/resources and CPU audit are
archived. Production audit copy: `profiles/session31-native-audit.json`.

Fetched main at start/end. Initial **ab141cd** was already merged. Late
**47bcae0** adds only Claude's completed judge results to NOTES.md and was merged.
Production SHA256 remains
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Native CUDA, Triton/Gluon, plans and dispatch are unchanged; session26 production
validation still applies. No redundant unchanged-production tests or experimental
imports were introduced.

### Mechanism and measured rejection

One portable eight-CTA cluster per KV head, each CTA computing256 keys with
FP8 WGMMA QK and FP16 PV. After matrix work completes, the CTA reuses tile shared
storage for FP32 partials; a cluster barrier publishes them. Each CTA's first
warp reduces one query head through distributed shared memory, then computes
RMSNorm/gate/BF16 output. A second cluster barrier protects remote-read lifetime.
Every key/value is attended; precision, new-cache writes and sliding RoPE/window
semantics are preserved. No atomics, completion counters or separate merge
launch. Host partial allocations remain as unused binding scaffolding.

Session2's cluster design used slower streaming attention; this is a distinct
single-tile test. An unfused256-key control uses the same attention body with
eight global partials and unchanged production Triton merge. Neither wrapper
is a proposed dispatch; both run broadly on supported sliding L1 shapes.
No Triton code/plan tuning, cluster-size or tile-width sweep.

18shapes,60adjacent AB/BA graph pairs per contrast, H100 microseconds:

| sliding L1 | production | cluster | paired delta | unfused | paired delta |
|---|---:|---:|---:|---:|---:|
|B1 80K page1|4.320|7.232|+2.928|6.528|+2.240|
|B2 80K page1|4.992|7.488|+2.496|7.136|+2.208|
|B8 80K page1|7.392|9.936|+2.608|9.408|+2.048|
|B16 80K page1|10.240|18.816|+8.512|12.928|+2.672|
|B32 80K page1|15.712|31.072|+15.312|23.536|+7.968|
|B1 80K page64|4.320|7.216|+2.848|6.400|+2.032|
|DP1 4096 page64 scale1/64|7.344|9.952|+2.640|8.912|+1.472|

Each contrast has its own parent times; table parent is the cluster contrast.
Paired deltas need not equal differences of medians. All18shapes lose with both
wrappers. No favorable-repeat selection or fitted cutoff. Full12shape benches
cluster1.826x/unfused1.883x against fixed SGLang/H200 times are experimental
results, not production regressions or H200 estimates.

The larger tile already loses before fusion; the cluster does not recover that
loss and costs more at larger batches. Cluster236/unfused223registers; both
65536dynamic+1024static shared bytes, zero local/stack allocation. SASS contains
real cluster arrive/wait operations. These do not isolate synchronization cost
from compiler/scheduling/residency effects. No NCU or achieved-occupancy claim.
Close this topology rather than extending into a cluster/tile micro-sweep.

### Validation

- Both wrappers16/16, each3actual-native/13fallback cases; each completes all12
  bench shapes. Initial cluster suite's16passes were followed by a harness-only
  AttributeError because the wrapper lacked `_plan`; exported the required hook
  without changing GPU code, then completed bench separately. Error log retained.
- **2160 adjacent graph pairs**, **162 independent output/cache checks** around
  capture/timing on18shapes: TP8/DP,pages1/64,scales1/1/64,soft/coherent/edges,
  contexts10–120K, batches1–32.
- **144 actual-native fresh graph replays**,72per wrapper, changed QKV/cache/
  routes/lengths/positions/cache slots; halved and empty contexts included.
- Fused-native isolated **memcheck0errors,synccheck0errors,racecheck0hazards/
  errors/warnings**.24fresh replays per tool, both pages/scales,contexts10/257/
  4096and empty contexts. No Triton loader kernels launched. This is fused-kernel
  sanitizer coverage, not an independently sanitized unfused wrapper.
- Custom drivers lock generation/reference/capture/replay, flush512MB before
  every timed graph and subtract empty replay cost. Standard suite/bench uses
  bench.py's timing-loop lock; no overlapping own GPU jobs. All processes finished.
- CPU audit reproduces generated sources, verifies exact gather/QK-column/
  conversion coverage, disjoint shared tile buffers, production identity,
  result counts/paired medians and sanitizer summaries. SASS EOF whitespace
  normalized only; no generated CUDA edits after testing.

### Board, budget and next cycle

Claude's scored **d0f30dc0d435455d** has completed, per its own updated NOTES/COORD:
16/16,**2.113x**, CI[2.081,2.156], new best **rank18**. Refreshed official board
shows2.1129x/rank18 and that job. No duplicate status polling or request. This is
main824e52a's source, not an exact check of current Codex production. Prior
Codex3e786 remains a different earlier source and cannot qualify currentSHA1bcf.
Do not attribute the independent aggregate difference to native code.

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 remains **5/8checks,
3/3scored**. Official account **8/10checks,4/10scored**, leaving2checks/6scored;
actual check limit10/day. No pending Codex jobs. Neither STOP file exists.

1. Read COORD/fetch main next cycle. CurrentSHA1bcf still lacks exact-source H200
   qualification; preserve shared checks while Codex's scored budget is exhausted.
   Main824e52a is already integrated despite Claude's older reminder to merge it.
2. Close single-tile256/cluster8 sliding fusion: both fused and unfused lose
   decisively. Existing closed native reduction/load/grid/attention families stay
   closed. Further work needs a distinct mechanism with measurable critical-path
   evidence; do not reopen these parameter sweeps.
3. Keep22ca912 separate. Finish clean on pushed codex, remove claim and post
   results. No judge job requires waiting.

## Session 32 — 2026-10-24, 03:37–03:46 UTC

Completed a focused **native sliding phase-latency diagnosis** to establish a
measured target after the rejected instruction/fusion families. **No production
kernel/dispatch change and no judge requests.** Experiment **3b0b2e9** is pushed
on **codex-exp/native-phase-latency**; start with
`experiments/phase_latency/README.md`. Frozen sources, generator, raw graph pairs,
per-warp cycle traces, fresh checks, SASS/resources, unmodified-production NCU
source/metric CSVs and CPU audit are archived. Production audit copy:
`profiles/session32-native-audit.json`.

Fetched main at start/end; **origin/main47bcae0** is already merged. Production
SHA256 remains
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Native CUDA, Triton/Gluon, plans and dispatch are unchanged; session26 production
validation still applies. No experimental imports or redundant unchanged-source
production suite were introduced.

### Probe design and findings

Each diagnostic kernel timestamps one phase with two volatile clock64 reads,
then writes each warp leader's pair to fresh, disjoint trace storage at the end.
Ten adjacent intervals, whole-body and consecutive-clock controls; a no-clock
variant compiles instrumentation away. No extra synchronization or omitted
computation. Complete wrapper preserves native shape predicate and PDL merge.
Two production graph captures provide an additional identical-code control.

Seven shapes initially, five second-seed/reversed-capture confirmations. Median
warp-zero cycles, initial/confirmation (warp zero holds all8 real query heads):

| interval | B1/80K/page1 | B8/80K/page1 |
|---|---:|---:|
| metadata/gather issue |1726/1611|1833/1875|
| query/new-KV preparation |2448/2401|5662/5573|
| residual copy wait |5/5|5/5|
| first CTA rendezvous |21/21|22/23|
| V conversion/publication |347/347|408/403|
| QK+wait |128/128|177/173|
| softmax/probability packing |657/656|769/770|
| PV+wait |182/182|251/247|
| whole body |5674/5697|9960/10100|

B1/page64 repeats the pattern. Other warps reach copy wait earlier and wait
~1530–1576cycles initially atB1/page1: gathering is overlapped, not free. Short
B1ctx10 behaves differently around the new-token owner: warp1 publication~850
cycles; warp0's V-publication interval~1140cycles. Raw owner/non-owner summaries
are retained. All phase readings are cycles, not converted kernel microseconds.

The front of the kernel is a more promising target than small matrix/conversion
loops. This is a broad latency diagnosis, **not a production speedup or proof
that the preparation duration is removable**. Separate compiled probes and
replays have different scheduling: do not sum their interval medians. Initial/
confirmation B1/page1 whole-body instrumentation graph deltas+0.272/+0.032us;
identical production capture offsets range-0.256 to+0.160us across shapes/runs.
No-clock SASS-identical control offsets reach+0.224/-0.176us. Capture/allocation
variation limits small timing claims. No H200 extrapolation or new cutoff.

SASS confirms all26page/phase variants' clock placements, including single copy
DEPBAR and complete QK/PV operations+waits. Consecutive-clock interval has no
intervening instruction and reports2cycles. No-clock control is instruction-for-
instruction identical to production on both pages (1056/1064sites). Diagnostic
kernels use74registers except QK interval75;17408shared,zero stack/local.
This control measures clock instruction overhead but cannot subtract away all
instrumentation-induced scheduling or late trace-store effects.

Three unmodified-production NCU launches after all timing/fresh work:
B1page1/B8page1/B1page64 long-scoreboard samples55/207/52 of176/595/167;
barrier22/88/21; executed warp instructions164482/1315856/167042. Achieved
active-warp percentages5.96/21.52/5.96. Small PC sample counts support broad
memory-dependency interpretation only, not exact stage percentages or isolated
cache latency. Profiler durations are not graph timings. Each profiled output/
cache independently passes reference.

### Validation and limits

- Diagnostic whole-body wrapper16/16 (3actual-native,13fallback), followed by
  all12bench shapes;1.982x versus fixed SGLang/H200 times is diagnostic evidence.
- **9520 adjacent AB/BA graph pairs**,540independent output/cache checks at
  comparison checkpoints. TP8/DP,pages1/64,scales1/1/64,soft/coherent/edges,
  contexts10/2047/4096/80K,B1/3/8. Separate **2208 cold trace replays**.
- **364actual-native fresh graph replays**,7shapes x13modes x4inputs. QKV/cache/
  routes/lengths/positions/cache slots change; shortened and empty contexts pass.
- Custom drivers lock generation/reference/capture/replay/profiling, flush512MB
  before every timed or traced replay, and subtract empty replay for graph times.
  Standard suite/bench retains bench.py's timing-loop lock; no own GPU overlap.
- No sanitizer run or new sanitizer claim. Synchronization is unchanged; CPU
  audit exhaustively checks trace indexing is in-bounds/injective for allS1–32
  and used batch/head shapes. Audit reproduces sources, verifies frozen production
  identity, SASS control/clock locations, raw pair medians and all result counts.
  Generated CUDA archive/SASS/resources received whitespace-only normalization
  after testing; no CUDA token change. All local processes completed.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 stays **5/8checks,
3/3scored**. Official account **8/10checks,4/10scored**,2checks/6scored remaining;
actual check cap10/day. Board best **2.1129x,rank18**, Claude's completedd0f30.
Current CodexSHA1bcf still lacks exact-source H200 qualification. No pending
Codex jobs and neither STOP file exists. Do not poll older resolved jobs.

1. Read COORD/fetch main; preserve scarce shared checks while Codex's scored
   allowance is exhausted. Claude currently owns Gluon consumer pipelining.
2. Further native work needs a distinct way to reduce metadata/preparation
   dependency or redundant work without a larger launch/synchronization cost.
   These measurements do not reopen session19 load-placement, session14L8
   preparation, padded-row softmax, reduction/PDL, cluster or tile sweeps.
3. Keep diagnostic3b0b2e9 separate. Finish clean on pushed codex, remove shared
   claim, record results, and stop. No judge job requires waiting.

## Session 33 — 2026-10-24, 03:47–04:09 UTC

Completed a focused **cooperative native sliding-query preparation** cycle.
**No production change and no judge requests.** All three implementations lose
paired complete-graph comparisons. Experiment **fda33df** is pushed separately
on **codex-exp/native-query-cooperative**; start with
`experiments/query_cooperative/README.md`. Sources/generator, raw pairs, fresh
checks, sanitizers, SASS/resources, NCU counters and reproducible CPU audit are
archived. Production audit copy: `profiles/session33-native-audit.json`.

Fetched main at start/end. Initial47bcae0 was already merged; final **2c681b3**
adds Claude session12 notes and completed judge profiles, and was merged into
codex. Production SHA256 remains
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Native implementations/dispatch, Triton/Gluon and plans are unchanged; session26
production validation still applies. No redundant unchanged-production suite or
experimental import was introduced. Claude now owns H200-bandwidth emulation
and full-layer Triton/Gluon planning; stay in the native lane.

### Mechanism and result

Session32's preparation interval motivated distributing the eight real query
heads over all four existing warps: sixteen lanes per head, four elements per
lane, FP32 RMS normalization/RoPE, FP8 E4M3 packing. Three complete variants:

- Linear compact512-byte shared-query handoff to the existing warp-zero WGMMA
  fragments, using an existing CTA barrier. Unlike session13's shared-query
  control, this stages no padded query rows and uses sixteen lanes per head.
- XOR-swizzled shared handoff: removes the linear layout's four-way read-bank
  conflicts without changing its producer/consumer topology.
- Two real query rows in each warp's WGMMA row block, using four warp-local
  shuffles to populate fragments. No shared-query handoff. Each warp writes two
  heads back to the original contiguous partial layout.

All preserve every key/value, FP8 cache/query limits, FP16 PV operands, FP32
partials, new-cache writes, BF16 output, PDL, three CTA barriers and launch count.
Normalization reduction order changes within FP32. No tile/split/dispatch sweep.

Sixteen shapes per variant,60 adjacent AB/BA pairs for candidate and identical-
production-capture contrasts. Confirmation reverses capture order with a second
seed and120pairs: four shapes for linear, eight each for swizzled/distributed.
H100 paired candidate-minus-production deltas, microseconds, initial/confirmation:

| sliding L1 | linear shared | swizzled shared | warp-local mapping |
|---|---:|---:|---:|
|B1,80K,page1 (scored)|+0.576/+0.448|+0.640/+0.416|+0.544/+0.352|
|B8,80K,page1|+0.448/+0.176|+0.400/+0.480|+0.496/+0.448|
|B1,80K,page64|+0.592/+0.704|+0.720/+0.640|+0.576/+0.608|
|DP1,4096,page64,scale1/64|+0.240/+0.320|+0.240/+0.192|+0.192/+0.096|

All16initial shapes lose in every variant. Identical-code capture offsets reach
-0.224/+0.160us across runs; small nearby deltas should not be overinterpreted.
Paired deltas need not equal differences of independent medians. Full12shape
bench aggregates linear1.974x/swizzled1.972x/distributed1.977x against fixed
SGLang/H200 numbers are experimental local results, not H200 speedups or a
production regression. No promoted cutoff or H200 request is justified.

### Evidence and validation

Static page1/EARLY SASS sites parent/linear/swizzled/distributed:
1056/896/912/888. All use74registers, zero stack/local allocation. Shared memory
17408bytes for parent/distributed,17920 for the shared variants. Three barriers,
two QGMMA and four HGMMA sites remain; distributed's four SHFL.IDX sites are
unpredicated. CPU audit checks all21 shared Python function ASTs, full packed-word
producer/consumer coverage, bank maps, and exact distributed fragment/output maps.

Four isolated B1/page1 NCU profiles, each with independent output/cache checks:
parent/linear/swizzled/distributed shared read-bank conflicts **0/768/0/0**;
shared write-bank conflicts all0. Executed warp instructions
**164482/169546/171402/185546**. HBM reads556032/551680/552448/551936bytes.
The swizzle fixes the measured bank conflicts, but neither it nor removing the
handoff produces a graph win. Lower static counts do not mean less executed work
when preparation and output handling involve more warps. No isolated latency,
achieved-occupancy or profiler-duration speedup claim.

- Three16/16 suites, each3actual-native/13fallback cases, each followed by all12
  benchmark shapes. Distributed test and bench logs are separate (see below).
- **10560 adjacent graph pairs**, **612 independent output/cache reference
  checks** at timing checkpoints, plus four profiler output/cache checks.
- **192 actual-native fresh graph replays**,64per variant,16shapes x4inputs:
  TP8/DP,pages1/64,scales1/1/64,soft/coherent/edges,contexts10–120K,B1–8.
  QKV/cache/routes/lengths/positions/cache slots change; shortened/empty contexts.
- Swizzled and distributed each independently pass **memcheck0errors,
  synccheck0errors,racecheck0hazards/errors/warnings**:24fresh native graph
  replays per tool/variant. Torch independently reduces FP32 partials and checks
  output/cache. No Triton loader launches; no separate linear sanitizer claim.
  EARLY=True tested; EARLY=False compiled only.
- Custom timing/fresh drivers hold the shared lock across generation/reference/
  capture/replay, flush512MB before every timed graph, subtract empty replay.
  Standard bench retains bench.py's timing-loop lock. All own processes finished.
- The shared GPU was held for an extended Claude emulation sweep. The initial
  distributed suite completed16tests and was intentionally stopped while waiting
  for the first timing lock; bench resumed separately. The swizzled NCU child was
  also waiting, without collecting counters, and was terminated. Its deferred
  log contains no measurements. Final profiling acquires the GPU lock in the
  parent **before starting NCU**, avoiding initialization ahead of lock ownership;
  the child skips a second lock acquisition. Initial successful parent/linear
  profiles used the earlier child-lock convention. No other agent was killed.
- Audit reproduces all generated sources, confirms unchanged production, counts
  validation records, recomputes paired medians and verifies resources/counters.
  SASS/resource text received whitespace-only normalization. Audit passes.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 stays **5/8checks,
3/3scored**. Final official account **8/10checks,4/10scored**,2checks/6scored left;
actual check cap10/day. Account best remains **2.1129x,rank18**, Claude's completed
scored d0f30dc0d435455d. Current CodexSHA1bcf still lacks exact-source H200
qualification. No pending Codex judge jobs; neither STOP file exists.

1. Read COORD/fetch main next cycle; preserve scarce checks while today's Codex
   scored allowance is exhausted. Claude's new emulation direction remains its
   owned work, not a Codex qualification or native benchmark.
2. Close cooperative single-tile query preparation, including compact shared,
   bank-swizzled and warp-local WGMMA-row mappings. The broad preparation interval
   from session32 is not enough to predict a profitable redistribution. Do not
   extend this into another lane/vector/layout/split sweep; earlier closed native
   reduction, load-placement, fusion, tile and full-attention families stay closed.
3. Any further native implementation needs a distinct measured dependency or
   redundant operation to remove. Keep fda33df separate; finish clean on pushed
   codex, remove the shared claim and post results. No judge job needs polling.

## Session 34 — 2026-10-24, 04:10–04:26 UTC

Completed native **computed RoPE** work and integrated Claude's late full-verify
merge improvement. Both native arithmetic candidates lose, so neither enters
production. **No judge requests.** Experiment **c8b85d5** is pushed separately
on **codex-exp/native-rope-compute**; start with
`experiments/rope_compute/README.md`. Sources, generator, arithmetic diagnostics,
raw graph pairs, fresh checks, sanitizers, SASS/resources, NCU and CPU audit are
archived. Production audit copy: `profiles/session34-native-audit.json`.

### Production integration

Initial main **2c681b3** was already merged. A late fetch found Claude's
**19b8ade**, with notes through **44e4e25**, merged into codex as **f1c18dc**.
This changes Gluon's full-verify merge to one-warp/two-head `_combine_hb` when
there are at most 32 partial slots. Native attention/reduction implementations
and their dispatch predicates are unchanged. Codex did not tune Triton/Gluon.

Final production SHA256:
`62fefe79bd913e6dc1b12388ad7bec50139096cb07c7b61918b8031c0c93a86b`.
The previous SHA1bcf source is frozen in `profiles/session34-previous.py`.
No experimental imports were introduced. The final source has **no exact-source
H200 qualification**, and no Codex scored allowance remains today.

Final integrated suite **16/16**, followed by all 12 bench shapes, **1.990x**
against fixed SGLang/H200 times; verify16/32 full **137.82/258.82 us**. This is
an unpaired local aggregate, not an H200 speedup estimate. The standard suite's
DP2/L8 full cases do exercise Gluon after its earlier dispatch expansion; extra
cases below cover larger TP8 batches and the new merge boundary too.

Ten complete-graph comparisons against the previous source, 80 adjacent AB/BA
pairs each, H100 microseconds:

| Full L8 case | previous | merged | paired delta |
|---|---:|---:|---:|
| B8, 80K, page1 (unchanged merge control) |80.800|80.960|-0.016|
| B12, 80K, page1 |111.424|108.736|-2.656|
| B16, 80K, page1 |140.096|138.560|-1.600|
| B32, 80K, page1 |256.320|255.520|-0.672|
| B64, 40K, page1 |256.864|254.912|-2.400|
| B16, 20K, page64, scale1/64, edges |49.872|47.648|-1.872|
| B24, 9K, page64, mixed |40.800|39.680|-1.152|
| DP2, 20K, page64, scale1/64 |50.736|48.880|-1.488|

Unchanged B1 native sliding/full controls have paired deltas 0.000/-0.240 us,
showing capture/allocation variation. Paired deltas need not equal differences
of separate medians. No universal or H200 speedup claim.

**800 adjacent pairs, 60 independent output/cache checks** at comparison
checkpoints. **80 fresh graph replays**, 40 per source across ten shapes, changing
QKV/cache/routes/lengths/positions/cache slots, including shortened and empty
contexts. Of the 40 merged replays, 28 exercise the new grouped Gluon merge,
4 the unchanged Gluon merge and 8 native-path controls. All pass. No new
integrated sanitizer claim; existing native kernels are unchanged. Drivers and
raw evidence: `profiles/session34-integration*.py` and `session34-integrated-*`.
CPU audit verifies all 20 shared helpers against main44e4e25, exact unchanged
native extension ASTs and `kernel_fn`, source hashes and complete result counts.

### Native computed RoPE: rejected

Session32's preparation interval motivated eliminating the position-dependent
RoPE-table address. Both variants load a 32-element frequency vector through
position-independent addresses and compute the same FP32 position-times-frequency
angle. Frequencies depend only on configuration/device, never input data.

- FP64 control: round turns and reduce the angle with an FP64 FMA before FP32
  hardware sine/cosine.
- Compensated FP32: two FP32 pieces of 2*pi and two explicit RN FMAs remove FP64
  arithmetic/conversions while preserving cancellation accuracy.

Both use PTX sine/cosine on a reduced angle near [-pi,pi], unlike the earlier
libdevice attempt. Every key/value, FP8 query/cache precision, FP16 PV operands,
FP32 partials, BF16 output, PDL, gathers, three barriers and native dispatch remain.
No query-layout/load-placement/tile sweep. The unused arithmetic probe exists
only in the experimental extension, never in production.

**201,326,592 angle comparisons per variant**: all 2^21 supported positions,
32 frequencies, theta500000/10000/1000000. Maximum absolute sine/cosine error
against Torch's FP32-angle results is **5.365e-7 / 7.600e-7**, FP64/compensated.
These are arithmetic errors, not output-relative errors or bit-exact FP8 claims.
Complete attention also passes at contexts 1,048,576 and 2,097,151.

18 shapes per variant, 60 adjacent AB/BA pairs per candidate/control contrast.
H100 candidate-minus-production paired deltas, microseconds:

| Sliding L1 case | FP64 | compensated FP32 |
|---|---:|---:|
| B1, 80K, page1 (scored) |+0.512|+0.448|
| B8, 80K, page1 |+0.864|+0.608|
| B1, 80K, page64 |+0.640|+0.512|
| DP1, 4096, page64, scale1/64 |+0.704|+0.512|
| B1, 2^21-1, page1 |+0.576|+0.512|

All 18 shapes lose for both variants. Identical-production capture controls
range -0.144 to +0.192 us. No favorable-repeat selection, fitted cutoff or
experimental H200 request. Broad complete-wrapper bench aggregates 1.978/1.969x
are experimental local results, not production regressions.

Page1/EARLY static sites parent/FP64/compensated **1056/1136/1096**. Each computed
variant removes 10 global load sites but adds 10 sine and 10 cosine sites plus
range reduction. Compensated has no FP64 instructions. All use **74 registers,
17408 shared bytes, zero stack/local allocation**, 2 QGMMA/4 HGMMA and 3 barriers.
Three isolated NCU profiles, independently reference-checked: executed warp
instructions **164482/172674/168578**; HBM reads **554752/555776/555008 bytes**;
shared read/write bank conflicts all zero. This demonstrates added work without
isolating its latency or attributing the entire graph loss to one instruction.
Profiler durations are not graph timings; no achieved-occupancy claim.

Validation: both **16/16** (3 native/13 fallback), each followed by all 12 bench
shapes. **4320 adjacent graph pairs, 324 comparison output/cache checks**;
**144 fresh native graph replays** across 18 shapes per variant, plus **24
alternate-config output/cache checks** (16 graph, 8 eager). TP8/DP, pages1/64,
scales1/1/64, soft/coherent/edges, contexts10–2^21-1, shortened/empty contexts.
Each variant independently passes native **memcheck0 errors, synccheck0 errors,
racecheck0 hazards/errors/warnings**, 24 fresh replays per tool. Torch independently
reduces partials and checks output/cache; no Triton loader kernels. EARLY=True
runtime coverage; EARLY=False compiled only. Source/count/SASS audit passes.

Custom drivers lock generation/reference/capture/replay, flush512MB before each
timed graph and subtract empty replay. Standard benches retain bench.py's timing
lock and run sequentially with own work. NCU's parent acquires the lock before
starting the profiler, after all native timing/fresh/sanitizer work. All processes
finished. Only generated disassembly/report whitespace was normalized.

### Budget and next cycle

Session **0 checks, 0 scored**, no withdrawals. Codex2026-10-24 remains
**5/8 checks, 3/3 scored**. Final official account **8/10 checks, 4/10 scored**,
leaving **2 checks/6 scored**; actual account check cap10/day. Board best remains
**2.1129x**, now **rank19**, Claude's completed d0f30 job. No pending Codex jobs;
resolved older jobs need no polling. Neither STOP file exists.

1. Read COORD/fetch main next cycle. Main44e4e25 is integrated. Current SHA62fefe
   needs its own exact-source H200 check before any future scored submission;
   neither main's older result nor any Codex older check qualifies it. Preserve
   scarce shared checks while Codex's scored allowance is exhausted.
2. Close computed native RoPE with these two reductions. Accurate arithmetic
   does not make the lookup replacement faster. Earlier closed native
   query-layout/load-placement/reduction/fusion/tile families stay closed.
3. Main session13's wide-gather full-decode observation is a possible distinct
   lead, not proof of a native win. Its H200 emulation is Claude's experiment,
   not an H200 measurement. Any native follow-up needs complete attention with
   low per-tile work, rather than another gather-only or large-single-tile sweep.
4. Keep c8b85d5 separate. Finish on clean pushed codex, remove the claim and post
   results. No judge job requires waiting.

## Session 35 — 2026-10-24, 04:27–04:39 UTC

Completed a focused **transposed native full-decode WGMMA** cycle. Neither
implementation wins on the scored full-decode cases. **No production change
and no judge requests.** Experiment **fded6bf** is pushed separately on
**codex-exp/native-transposed-decode**; start with
`experiments/transposed_decode/README.md`. CUDA sources/generator, frozen
production, complete graph pairs, fresh checks, sanitizers, SASS/resources,
NCU and reproducible CPU audit are archived. Production audit copy:
`profiles/session35-native-audit.json`.

Initial main44e4e25 was already merged. Final fetch found **a8de0e7**, Claude's
session14 notes only, and merged it into codex. Read its sliding fixed-cost and
FP8 P@V observations; no Triton/Gluon tuning by Codex. Production remains SHA256
`62fefe79bd913e6dc1b12388ad7bec50139096cb07c7b61918b8031c0c93a86b`.
No native implementation/dispatch or inherited helper/plan changed. Session34's
integrated production validation still applies; no redundant unchanged-source
suite or new production sanitizer claim. No experimental imports were added.

### Mechanism and results

One128-thread CTA handles a KV-head pair with a three-stage ring of64-key,
full128-byte-row K/V gathers. Per head, QK becomes K[64,64] @ Q.T[64,8], FP8
m64n8k32, and PV becomes V.T[64,64] @ P[64,8], FP16 m64n8k16. Each thread holds
only four score/output floats per head. This removes padded matrix work but
requires cross-warp max/sum reductions and shared probability staging.
It is a streaming design, unlike session28's large single-tile grid; main's
previous transposed experiment used warp MMA rather than this WGMMA layout.

The first version explicitly transposes V while converting it to FP16. The
follow-up keeps V's natural layout with packed conversion/vector stores and
uses WGMMA transA=1. This sole conversion/descriptor/MMA change is generated
reproducibly. Both preserve every key/value, FP8 query/cache limits, FP16 PV,
FP32 online softmax/partials, cache insertion, BF16 output and inherited PDL
merge. One fixed grid policy, ceil(528/pairs), capped128 splits; no stage/tile/
warp/split sweep or fitted dispatch cutoff.

Eight shapes per variant,60 adjacent AB/BA complete-graph pairs, H100 us:

| Full L1 case | production / explicit transpose | paired delta | production / natural V | paired delta |
|---|---:|---:|---:|---:|
|B1,80K,page1|16.720 / 34.240|+17.600|17.056 / 22.112|+5.072|
|B16,80K,page1|130.016 / 308.880|+178.816|129.856 / 185.296|+55.600|
|B32,80K,page1|241.888 / 570.544|+328.560|243.200 / 337.728|+94.672|
|B1,80K,page64|16.592 / 31.392|+14.720|16.480 / 19.072|+2.560|
|B4,6K,page64,scale1/64,edges|11.120 / 12.240|+1.136|11.456 / 10.288|-1.184|
|B2,9K,page64,mixed,soft|13.040 / 7.600|-5.472|13.440 / 5.984|-7.408|
|DP1,4096,page64,scale1/64|11.328 / 15.248|+3.824|11.584 / 12.096|+0.528|

Each variant has its own paired production capture. Paired deltas need not
match differences of independent medians. Scored losses are decisive; no
favorable-repeat selection, new cutoff or H200 request is justified. Nearby
short-context gains do not reopen session29's closed short-context audit.
Whole-wrapper bench1.440/1.750x against fixed SGLang/H200 times is experimental
local evidence, not production regression or H200 speedup.

### Evidence and validation

Page1/page64 register counts: explicit transpose120/113, natural V94/80;
both69,632 dynamic shared bytes (structure-layout audit) plus1,024 static bytes
reported by cuobjdump, no local/stack allocation. Each page has4QGMMA/8HGMMA
static sites. Both have identical matrix dimensions and precision.

Two isolated B16/80K/page1 NCU launches, independently reference-checked:
executed warp instructions63,602,660 /48,439,316; shared store bank conflicts
17,024,174 /1,269,607; shared read conflicts1,220 /1,677; HBM reads
335,207,936 /335,176,704 bytes. Short-scoreboard stalls48.67/22.46%, long-
scoreboard13.46/26.98%, barrier3.00/5.80%. Removing the software transpose
removes substantial work/conflicts, but does not make complete attention
competitive. These counters do not isolate latency or attribute remaining
conflicts to a specific instruction. No profiler-duration/occupancy speedup claim.

- Both wrappers **16/16** (5actual-native/11fallback), each followed by all12
  benchmark shapes. Also16 boundary smoke checks per variant, pages1/64,
  contexts0/1/10/63/64/65/257/4096.
- **960 adjacent graph pairs**, **96 independent output/cache checks** at
  comparison checkpoints across eight cases per variant.
- **64 actual-native fresh graph replays**, plus64production controls:
  four inputs/case, changing QKV/cache/routes/lengths/positions/cache slots,
  halved/empty contexts, TP8/DP,pages1/64,scales1/1/64,coherent/soft/edges.
- Natural-V variant independently passes **memcheck0errors, synccheck0errors,
  racecheck0hazards/errors/warnings**,24fresh native replays per tool.
  Torch reduces FP32 partials and validates output/cache; no Triton loader
  launches. No separate sanitizer claim for the explicit-transpose variant.
- Custom drivers lock generation/reference/capture/replay, flush512MB before
  every timed graph, subtract empty replay. Suite tests hold the lock; standard
  bench uses its timing-loop lock. All own GPU jobs are sequential. NCU parent
  takes the lock before launching the profiler, after timing/fresh/sanitizers.
- CPU audit reproduces natural source, checks frozen production identity,
  fragment/gather/swizzle/chunk coverage, raw medians and complete result counts.
  Generated disassembly whitespace only was normalized. All processes finished.

### Budget and next cycle

Session **0checks,0scored**, no withdrawals. Codex2026-10-24 remains
**5/8checks,3/3scored**. Final official account **8/10checks,4/10scored**,
2checks/6scored left; actual account check cap10/day. Best remains **2.1129x,
rank20**, Claude's completed d0f30 job. Current SHA62fefe still lacks exact-source
H200 qualification. No pending Codex jobs; neither STOP file exists.

1. Read COORD/fetch main. Maina8de0e7 notes are integrated; preserve scarce
   checks while Codex's scored allowance is exhausted. Don't poll resolved jobs.
2. Close this cooperative transposed WGMMA topology as a scored direction.
   Smaller matrix fragments do not offset its data movement and reduction costs.
   Another tile/stage/grid sweep is unjustified; future native work needs a
   concrete reduction in the remaining per-tile work or synchronization.
3. Claude's session14 sliding multi-warp partial reduction suggestion is Gluon
   work in its lane. Do not independently tune its plans or FP8 P@V experiments.
4. Keep fded6bf separate. Finish clean on pushed codex, remove the shared claim,
   post results, and stop. No judge job needs waiting.
