Session 11: native raw-cache shared-memory layout
================================================

Experimental only; production submission.py and Triton plans are unchanged.
Base production SHA256:
a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1
Base experiment baseline.py is session10 swizzle.py, with only build identifiers
changed: two producer warps, two consumer warpgroups, FP8 QK WGMMA, SW128 FP16 PV.

The initial instruction-level NCU SourceCounters profile (source-baseline.csv)
shows two independent opportunities. Each LDS.64 raw-V conversion load has
1,292,576 shared wavefronts vs646,288 ideal; the two loads account for1,292,576
excess wavefronts. QK WGMMA itself has zero excess shared wavefronts. The four
producer K LDGSTS instructions each report2,584,128 wavefronts vs323,016 ideal.
These are NCU source-attributed wavefront counts; do not confuse their totals
with the separate whole-kernel hardware LSU bank-conflict counters. The QK
consumer is not the evidence for changing K; the producer gather destinations are.

Variants generated from baseline.py by generate.py:
- rawv: remove the 16-byte-per-row padding from raw FP8 V addresses. Retain the
  same allocation sizes and FP16 PV layout to isolate the changed accesses.
- key64: put each raw FP8 K head in a K-major SW64 layout. Byte index i maps to
  i XOR (((i >>7)&3)<<4). Base alignment512 bytes, descriptor SBO512 bytes,
  LBO16 (assumed), swizzle2, second K32 operand starts at+32. Padded V unchanged.
- rawvk64: combine those independent changes, still the same allocation sizes.
- paired128: combine both K heads in one SW128 tile, each physical row128 bytes.
  Index=row*128+head*64+column, XOR (((index>>7)&7)<<4). Descriptor base selects
  head*64, then+32 for the second K32; SBO1024, swizzle1. Separate compact V
  arrays. This removes unused raw scratch as a consequence of the layout.

All variants preserve every-key attention, FP8 cache and query rounding,
FP16 P/V operands, FP32 accumulators/partials, normalization/gating, new-token
causality/cache writes, and fresh-input CUDA graph behavior. No new precision
approximation, host sync, data-value dispatch or cached input-derived state.
Runtime comparisons use stages4 and the same existing native split heuristic.
Other stages compiled by the extension have no new runtime validation claim.

The K-major canonical layouts and descriptor offsets follow NVIDIA PTX:
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor

Reproduce from repository root using /home/daytona/okc/bin/python:
  experiments/key_layout/generate.py
  experiments/key_layout/compare.py
  experiments/key_layout/compare_paired.py
  experiments/key_layout/finish_validation.py
  experiments/key_layout/sass.py

Comparisons perform independent reference checks before/after graph capture.
Graphs use cold L2 and subtract a trivial graph replay. Sixty rounds rotate and
reverse variant order; within-round native deltas are retained. This is not the
judge's balanced adjacent-pair estimator. Do not compare separate runs to infer
sub-percent effects. All generation, capture, correctness and timed GPU work in
the comparison drivers holds the shared GPU lock. Validation/profiling drivers
are serialized; no NCU/sanitizer overlaps our own timing/correctness work.

Measurements (H100, microseconds)
--------------------------------

compare:
variant                   B1           B16           B32
production                 18.304      155.056      306.512
baseline                   34.768      205.616      394.096
rawv                       33.920      197.744      377.088
key64                      33.472      195.168      371.152
rawvk64                    33.248      190.704      362.848

compare-paired:
variant                   B1           B16           B32
production                 18.432      154.336      308.112
baseline                   34.768      205.760      395.216
rawvk64                    32.640      190.384      362.688
paired128                  32.880      192.016      366.592

The combined SW64/compact-V variant saves7-8% versus the prior best native
prototype at large batches, but still takes18-23% longer than production.
The paired SW128 variant is slightly slower despite lower shared-memory use.
No production dispatch or new judge job is justified.

Compiled stage4/page1 resources from cuobjdump (resources.json):
  baseline92 registers, rawv90, key6494, rawvk6489:62528 shared bytes each;
  paired12888 registers,54336 shared bytes. Stack/local allocations zero.
ptxas reports 1024 fewer shared bytes in each case (61504/53312); use the loaded allocation for residency comparisons.
All still allow only two resident CTAs/SM at these static register allocations.
No new FP8 conversion, matrix-operation or vector load/store count reduction
is claimed: static counts for those instructions are unchanged by these layouts.

Validation completed before profiling:
- 27 variant/shape entries across two comparison drivers, including repeated
  controls. Independent output/cache reference checks before and after capture.
- rawvk64 and paired128 each pass16/16 full tests:five native,eleven fallback.
- Their full native/fallback bench aggregates are1.717x/1.706x versus fixed
  SGLang/H200 reference times, not production regressions or judge results.
- 84 all-native fresh graph replays,42 per retained variant, seven nearby
  cases each:TP8/DP,pages1/64,scales1/1/64,contexts10-120K,soft/coherent/edge,
  changed QKV/cache/routes/lengths/positions/cache slots, including empty context.
- rawv and key64 have scored comparisons only; no full-suite/fresh/sanitizer
  coverage is claimed for these isolated-component controls.
- Source generator reproduces all four archived variants byte-for-byte.

Native-only Compute Sanitizer completed for rawvk64 and paired128: memcheck0
errors, synccheck0 errors, racecheck0 hazards/errors/warnings. Each tool executes
24 native calls (two variants, four page/scale cases, three changed-QKV calls),
reduces FP32 partials independently in torch, and checks output/cache against the
reference. No Triton loader is involved in the isolated sanitizer path.

Next hypothesis, if another native cycle is warranted: this pipeline converts
and stores V BEFORE launching QK, then immediately waits for QK. Consider issuing
QK before that independent V conversion, waiting only before using scores. This
would target a specific serialized dependency rather than another stage/register
sweep. Verify generic/async-proxy ordering for new-token K, all warpgroup rules,
and fresh-input race safety. It is distinct from previously rejected deferred-PV
and early raw-slot release. No implementation or performance claim is made here.

Final isolated profiles (B16 full verify,16 splits,stage4,page1)
-----------------------------------------------------------
The diagnostic baseline source profile ran before experiments. All final
profiles ran sequentially after correctness/timing/sanitizer work under the
shared GPU lock. Source profiles and hardware metrics are separate
passes. No profiler duration is used as a graph timing.

                     baseline       rawv      key64    rawvk64   paired128
hardware load
bank conflicts        1292621         31    1292613         39          25
hardware store
bank conflicts        1306878    1348653     852156     760364      657703
achieved occupancy%      29.19      29.30      28.78      28.97       28.79
issued warps/scheduler     .54        .56        .59        .58         .59

Instruction-attributed raw-V excess shared wavefronts fall1292576->0 when
padding is removed. Aggregate cache-gather excess shared wavefronts fall
12597624->2584128 with rawvk64 and->1292064 with paired128. The ideal cache-gather
count is2584128 for each. QK WGMMA has zero excess in every version.
profile-summary.json distinguishes these source counts from hardware counters.
The paired layout improves those counters further, but does not improve graph
timing; shared conflicts alone do not explain the remaining production gap.

Cycle uses0 checks/0 scored runs. Codex2026-10-23 totals3/8 checks,2/3 scored;
account12/20 checks,7/10 scored at final read. Best remains2.020924x, rank22.
All local subprocesses completed successfully. Production source is unchanged.
