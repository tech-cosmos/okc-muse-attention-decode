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


## Session 4 — 2026-10-23, 22:07 UTC onward (in progress)

Candidate **fddd5c1**, branch `codex-exp/cuda-reduce`: eight-warp hierarchical
native FP32 reduction for full layers with B*Hkv<=4 and 64<S<=128. Main 651b421
merged; its one-pass sliding merge beat the old native sliding path by about
0.3–0.4 us, so the old native sliding code/dispatch was removed. Final local
suite 16/16; full bench 1.882x versus fixed SGLang/H200 times. Broader native
hierarchies, tiled reductions, cluster/DSM reductions and specialized indexing
were investigated; only the small full-batch hierarchy is retained.

Check **40e1aa41415a4c87**, code fddd5c1, requested about 22:19 UTC: pending.
Codex UTC-day usage is now **2/8 checks, 1/3 scored runs**. Do not submit unless
this exact code's check beats the account's board best at submission time.
The preceding account read showed 9/20 checks, 5/10 scored (before our new check).
