Session 10: native warpgroup PV with swizzled shared values
=========================================================

Experimental only. No production imports or dispatch changes. Base production
submission.py is SHA256 a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1,
the checked/scored 5e2441f/effc2bf source, incorporating main65ff1ab. origin/main
468a53d has already been merged and changes only notes/profiles since main65.

baseline.py is session9's forced.py (vector V staging and x4 matrix loads), copied
from codex-exp/cuda-matrix-loads. ancestor_v9.py is the session1 single-head WGMMA
PV source, used only as input to generate.py; it is never imported by any driver.
The new test combines register-input FP16 WGMMA PV with paired-head full-row
hardware-barrier gathers and vectorized V conversion. It does not tune Triton.

Variants generated deterministically by generate.py:
- sync: replace eight LDSM.x4 and sixteen HMMA.16816 with two HGMMA.64x64x16.
  Register P uses the existing packed half2 layout; output remains FP32. V uses
  8x8 blocked N-major shared storage with no swizzle, and generic/async fences.
- overlap: alternate two V buffers and defer PV completion to the next QK wait.
  Final loop exit explicitly drains pending PV; compiler accumulator fences keep
  reads after completion. SASS confirms no end-of-iteration PV wait.
- swizzle: same synchronous PV with N-major SW128 V, aligned to 1024 bytes.
  For half-element index i, target = i XOR (((i >> 6) & 7) << 3). The descriptor
  has zero base offset, LBO/SBO=1024 bytes and swizzle mode1. Full vector stores
  are retained. This addresses the additional bank conflicts from blocked V.
- swizzle_overlap: apply the same swizzle to alternate-buffer deferred PV.
- early_release: start from swizzle; publish the raw-cache ring slot immediately
  after QK completion, before softmax/PV, and remove the final consumer barrier.
  Raw V was already converted before the prior barrier. PV reads separate V
  storage protected by its completion wait, not the released raw storage.

Every variant attends every key, preserves FP8 cache/query rounding, exact FP8
value conversion to FP16, FP16 probabilities, FP32 accumulators and partials,
causal new-token attention and cache writes. All data-dependent work remains
inside each call. Same stages4 and native split rule across all comparisons.

NVIDIA PTX documentation consulted for register fragments, transpose/layout,
shared descriptors, async fences and input/accumulator completion requirements:
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instruction-wgmma-mma
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group

Measured H100 cold-L2 CUDA graph medians (microseconds)
----------------------------------------------------
Each comparison captures all graphs once, rotates/reverses order over 60 rounds,
subtracts an empty replay, and holds /tmp/okc-gpu.lock through capture/reference
checks/timing. It is not the judge's balanced adjacent-pair estimator. Interpret
large gaps; do not infer a small gain by comparing separate runs. Raw times and
within-round deltas are in results/compare*.json.

                          B1            B16            B32
compare.json:
 production              18.464        154.912        304.880
 vector native baseline  35.440        211.600        402.032
 sync                    36.592        264.896        515.248
 overlap                 36.720        270.512        525.392
compare-swizzle.json:
 production              18.464        156.160        303.392
 vector native baseline  34.656        212.528        399.120
 swizzle                 34.336        207.136        390.256
 swizzle_overlap         34.432        210.336        396.496
compare-early.json:
 production              18.432        154.928        302.464
 vector native baseline  34.544        211.056        397.584
 swizzle                 34.208        205.552        388.768
 early_release           34.288        206.272        391.040

Swizzling is necessary to make this WGMMA PV design competitive with the native
baseline. The best variant saves about 2-3% at large batches but still loses to
production by 29-33%. Deferred PV and early raw publication do not help. Keep the
production path. No experimental judge check or scored run is justified.

Stage4/page1 compiled resources: baseline94 registers,62528 shared bytes;
sync94,61504; overlap95,69696; swizzle92,61504; early_release92,61504.
No spills in these builds. The native design still fits two CTAs/SM. Static
SASS is archived, and sass-counts.json counts full-function instructions, not
executed instructions. All new PV versions replace the baseline eight LDSM and
sixteen HMMA with two HGMMA, preserving eight FP8-pair conversions per tile.

Validation
----------
- 36 variant/shape entries across three comparison scripts, including repeated
  controls. Independent reference checks before and after each graph capture.
- swizzle and early_release each pass16/16 full tests: five native full-verify
  cases, eleven production fallback cases. Runtime stages4 only.
- Both native/fallback full benches report1.657x versus fixed SGLang/H200 times;
  these are experimental H100 aggregates, not production regressions or judge runs.
- 84 all-native fresh graph replays,42 per retained version, across seven nearby
  shapes: TP8/DP, pages1/64, scales1/1/64, contexts10-120K, soft/coherent/edge.
  Changed QKV/cache/routing/lengths/positions/cache slots, including empty contexts.
- sync, overlap and swizzle_overlap have the scored-shape reference/capture checks
  above only; no full-suite/fresh/sanitizer claim for those discarded variants.

Reproduction from repository root, using /home/daytona/okc/bin/python:
 experiments/pv_warpgroup/generate.py
 experiments/pv_warpgroup/compare.py
 experiments/pv_warpgroup/compare_swizzle.py
 experiments/pv_warpgroup/compare_early.py
 experiments/pv_warpgroup/finish_validation.py
 experiments/pv_warpgroup/sass.py
The last validation driver serializes full tests, benches, fresh replays,
sanitizers and profiles. Each native profiling/sanitizer driver holds the same
GPU lock. No profiling overlaps this session's other local GPU work. Build
caches are omitted from git. Two launchers briefly shared the initial compilation;
the duplicate was terminated before any GPU work, and its stale completed-build
locks removed. The retained comparison process completed normally.

Production scored job a48b8c849e53426b resolved this session:16/16,2.011402x,
CI[1.981231,2.041573], no flags. Main's best remains2.020924x, rank21 at session
start. Native target H200 decode1/verify1 full14.922/16.670 us versus main scored
14.997/16.908, independent runs with overlapping aggregate confidence intervals.
The existing native small-full merge remains; these runs don't isolate its gain.
No new checks/scored runs this cycle; Codex day totals3 checks/2 scored.

Next direction, if another native cycle is warranted: investigate the untouched
raw-K producer/QK shared layout with measured bank-conflict evidence first. It
still uses unswizzled 8-row blocks, whereas V now has an efficient SW128 layout.
Do not assume another swizzle will close the remaining gap, and do not repeat
stage/register/deferred-PV sweeps without a distinct mechanism. The synchronous
swizzle.py is the strongest building block from this cycle; early publication
and double buffering do not improve it on the scored full-verify shapes.

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
