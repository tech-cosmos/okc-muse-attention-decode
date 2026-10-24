Session12: native QK / V-conversion overlap (experimental only)
==============================================================

This folder is not imported by submission.py. Production is unchanged. The
starting kernel is session11 rawvk64.py from codex-exp/cuda-key-layout157c4d0,
renamed baseline.py with only build/module names changed.

Hypothesis
----------
Previously each tile converted/staged V, fenced and synchronized, issued two
FP8 QK WGMMA operations, then immediately waited for scores. Issue QK first and
perform independent V conversion while it runs. Keep FP8 queries/cache, FP16 PV
operands, FP32 partials, all keys, normalization/cache writes and the split rule.
All measured/validated variants use stages 4; other compiled stages are not
claimed runtime-validated this session. No Triton plans or kernels are tuned.

Variants
--------
overlap: preserve the original pre-QK proxy fence and consumer barrier; commit
QK, convert V, fence/synchronize V, then wait for QK scores.
latebar: same, but put V's consumer barrier after softmax, just before PV.
nobar: overlap without the redundant pre-QK consumer barrier.
latebar_nobar: latebar without that same pre-QK barrier.

Ordering argument
-----------------
Every consumer waits with acquire on the ready mbarrier. For cached tiles it
tracks all 64 producers' cp.async completion; for new tokens the producers keep
their existing generic-store barrier, proxy fence, and release arrivals.
Each consumer keeps a K proxy fence before QK. No consumer modifies K, and
all consumers acquire the same completed producer phase, so publishing
consumer-owned shared data is not a role of the removed pre-QK barrier.

Score accumulators are not accessed until wgmma.wait_group 0 and the existing
compiler-register fence. QK reads raw K while conversion reads distinct raw V
and writes a separate FP16 V buffer. All V stores precede each thread's proxy
fence and the group publication barrier before PV. PV still completes before
the consumer end barrier and empty-slot release; no raw buffer is recycled
while either QK or conversion can read it. No deferred PV or early release.
The warpgroup branch and tile loop are uniform within each 128-thread group.

Primary ordering reference consulted:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
and the Async Proxy / wgmma.wait_group sections in that same PTX manual.

Reproduction (from repository root)
----------------------------------
. ~/okc/bin/activate
python experiments/qk_overlap/generate.py
python experiments/qk_overlap/compare.py
python experiments/qk_overlap/compare_nobar.py
python experiments/qk_overlap/finish_validation.py
python experiments/qk_overlap/sass.py
python experiments/qk_overlap/resources.py
python experiments/qk_overlap/schedule.py

finish_validation serializes all four 16-test suites and full native/fallback
benches, then fresh replays for nobar and latebar_nobar, isolated native
sanitizers, final comparison, and isolated profiling. Each native/fallback suite
has five native and eleven fallback tests. Each retained variant gets 42 fresh
all-native graph replays across seven shapes, including TP8/DP, pages 1/64,
scales 1 and 1/64, contexts 10-120K, soft/coherent/edge, changed QKV/cache/routing/
lengths/positions/cache slots, including empty contexts. Each sanitizer runs 24
native calls over two variants/four page-scale cases/three changed-QKV calls;
its torch reduction checks final output and cache independently of Triton.

Timing
------
Cold 512 MB L2 flush, CUDA graph replay, empty replay subtraction. 60 rounds
rotate/reverse variant order; each output/cache checked before/after capture.
GPU flock covers generation, capture, reference and all timed replay loops.
These are H100 local numbers, not judge H200 results or the judge's balanced
adjacent-pair estimator. Do not infer small effects across runs. Use the final
comparison as the authoritative all-variant measurement: compare_nobar was
started alongside the validation driver; initial reference tests serialized on
the GPU lock but bench.py warmups could overlap its final shape. That initial
comparison is exploratory only. Final comparison and profiling run after all
other local correctness/bench/sanitizer processes finish.

Static evidence
---------------
schedule-summary.json archives relevant SASS ordering. QGMMA now precedes the
V stores and the QK DEPBAR follows them; the first LDS/conversions can be
scheduled among QK instructions. This demonstrates instruction overlap, not a
runtime win. Matrix/conversion/vector-store counts remain unchanged. Stage4
registers 89/91/93/91/93 for baseline/overlap/latebar/nobar/latebar_nobar;
cuobjdump 62528 shared bytes (ptxas 61504), no spills, two resident CTAs/SM.
Full static dumps, resources and operation counts are adjacent to this README.

Validation outcome
------------------
All four variants pass 16/16 full reference tests (five native each). Nobar and
latebar_nobar each pass 42 fresh all-native graph replays. Isolated native
memcheck and synccheck report 0 errors; racecheck reports 0 hazards/errors/warnings.
The two extra-barrier controls have full-suite and scored/capture validation,
but no fresh-input or sanitizer claim this session. All runtime coverage is stages 4.
Full native/fallback bench aggregates: overlap 1.702x, latebar 1.683x, nobar 1.685x,
latebar_nobar 1.693x against fixed SGLang/H200 numbers. These are local rejected
experiments, not production regressions or judge measurements.

Final isolated comparison (microseconds, H100 cold-L2 graph)
----------------------------------------------------------
variant                 B1        B16        B32
production               18.176    154.304    303.856
baseline                 33.072    190.240    358.608
overlap                  32.640    193.504    366.128
latebar                  32.960    195.920    370.144
nobar                    33.200    191.152    360.576
latebar_nobar            33.280    193.904    367.408

No variant improves the prior native baseline at the target large batches. Removing the pre-QK barrier
recovers most of the added-barrier regression, but the final within-round
nobar-minus-baseline medians at B16/B32 are +1.216/+2.240 us. The native kernel
still trails production by about 19-24% on these shapes. Keep session11 rawvk64
as the better native building block; no production attention dispatch or judge
check is justified. Close this QK/V overlap direction and its barrier variants.

Isolated NCU result
-------------------
Baseline/nobar/latebar_nobar occupancy: 28.97/28.53/28.73%; issue rate 0.58 for
all three. Hardware shared-load conflicts 19/25/21, shared-store conflicts
747579/766737/688001. Source-attributed raw-V and QK excess shared wavefronts
are zero in all three; gather excess is unchanged at 2584128. These static
scheduling changes do not improve useful occupancy/issue throughput. Counter
collection is sequential after final comparisons; no profiler duration is
used as graph timing. summary.json captures the full validation/results audit.

Session budget: zero checks, zero scored, zero withdrawals. The UTC day changed
while wrapping up; account limits reset and production remained exact 5e2441f/
effc2bf. No pending Codex judge jobs and no production changes.
