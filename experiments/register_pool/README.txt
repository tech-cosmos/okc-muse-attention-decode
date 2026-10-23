Session 8: native register-pool experiment (2026-10-23)
====================================================
Rejected for production. No file here is imported by submission.py or sent to
okc. Production is the unchanged exact source checked by 01bc4ecc9c7047d9 and
scored as a48b8c849e53426b (pending at the time of these local experiments).

Hypothesis
----------
The previous paired-head native attention has 320 threads, 94 registers/thread,
and 62,528 shared bytes at stage4. Two CTAs fit per SM. Use Hopper's register
pool to give the gather warpgroup fewer registers than the two consumer groups,
allowing three CTAs with the same attention tile and shared-memory ring.

The producer group is padded to a complete four-warp group, and ready barriers
and gather strides account for all 128 producer threads. All keys, FP8 cache and
queries, FP16 PV operands, FP32 softmax/partials, new-token writes, and in-call
output reduction are retained. The grid uses 264 or 396 target CTAs for two or
three resident blocks, based only on shapes. No input-derived state is cached.

NVIDIA's setmaxnreg documentation requires warpgroup-uniform execution and
returns registers to a CTA-local pool. We use balanced totals: initial 56 each
-> consumer 72/72 + producer 24, or initial 80 -> 96/96 + 48. The emitted SASS,
not just PTX source, was checked. Reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg

Sources / compile findings
--------------------------
generate.py reproduces every source from baseline.py (session3 pair_mbar.py).
- dynamic.py: 384 threads, minimum 3 blocks, early 24/72 register change before
  common prologue. ptxas C7507 discards hints; ordinary 56-register code spills.
- static.py: same 384-thread/3-block topology without hints. Same bad timings.
- pool3.py: early 40/64 limits also discarded (compile-only).
- reload3.py: early 40/64 plus volatile shared query reload instead of eight
  query registers live across the loop; hints discarded (compile-only).
- pool2.py: early 48/96 hints DO survive, but huge spills (308/320 bytes static
  spill stores/loads for page1/stage4) and much worse performance.
- static2.py: 384 threads, minimum 2 blocks, no repartition; measured control.
- late3.py: put 24/72 repartition AFTER the common prologue barrier, in disjoint
  producer/consumer paths. Hints survive, initial 56 registers, three blocks/SM.
  page1/stage4: 96-byte stack, 104 bytes spill stores and 104 bytes spill loads.
- late2.py: late 48/96 repartition, initial 80 registers, two blocks/SM.
  page1/stage4: 32-byte stack/spill stores/spill loads; page64 has no spills.

The failed early layouts do not establish a general compiler restriction on
register redistribution: late3 successfully emits USETMAXREG at the same 24/72
limits. Placement relative to the common prologue changes the outcome here.
Build logs and stage4/page1 SASS excerpts are archived. Static spill byte counts
are compiler reports, not measured dynamic traffic.

Timing
------
H100 cold-L2 CUDA graph medians, 60 replays, empty replay subtracted, shared GPU
lock. Same-process sequential comparisons, NOT a balanced paired estimator.
Every measured implementation first passed an independent reference comparison.
Compare-pools.json (stage4):
                        verify1       verify16      verify32 (us)
production               18.480        155.552       304.352
old native               34.800        230.080       442.112
early pool2              57.072        513.824      1003.840
static2                  36.704        238.080       465.280
late2                    37.760        247.184       486.080
late3                    39.296        236.256       455.440

The three-block design nearly recovers the old native throughput despite lower
consumer register budgets, but never beats it or production. Do not adopt it.
Compare.json separately measures the discarded-hint 3-block variants: roughly
599-611 / 1122-1168 us at B16/32, similar to static 56-register controls.
Native/fallback full benches after full correctness: 1.548x late2 / 1.565x late3
against fixed SGLang/H200 times. These are experimental results, not production
regressions or judge scores.

Validation
----------
- 36 independent reference comparisons across the two timing drivers (includes
  production/baseline/control repetitions, not 36 distinct configurations).
- late2 and late3 each pass the full 16-test suite: 5 native, 11 fallback cases.
- 42 all-native fresh graph replays per variant (84 total), across seven nearby
  shapes. Pages1/64, TP8/DP, scales1/1/64, contexts10-120K; soft/coherent/edge data;
  fresh cache/QKV/routes/lengths/positions/cache slots, including empty contexts.
- Sanitizers and final NCU details are recorded in their logs and below when done.
- pool3 and reload3 are compile-only; no correctness/performance claim for them.

Reproduce (from repo root, with ~/okc activated)
----------------------------------------------
python experiments/register_pool/generate.py
python experiments/register_pool/compare.py
python experiments/register_pool/compare_pools.py
python experiments/register_pool/validate.py
VARIANT=late3 python experiments/register_pool/validate.py
python experiments/register_pool/fresh.py
VARIANT=late3 python experiments/register_pool/fresh.py
python experiments/register_pool/bench_all.py
VARIANT=late3 python experiments/register_pool/bench_all.py
python experiments/register_pool/run_sanitizers.py
# Only after every benchmark/validation/sanitizer process is finished:
python experiments/register_pool/run_profiles.py

Final validation/profile results
--------------------------------
Both retained variants: memcheck 0 errors; synccheck 0 errors; racecheck 0 hazards,
0 errors, 0 warnings. Each tool executed 24 native calls across both variants,
with pages1/64, scales1/1/64 and three changed-QKV inputs per case. The sanitizer
script independently reduces FP32 native partials in torch and checks final
outputs and cache writes against reference.

All benchmarks and fresh/reference tests exited before profiling began. The
profile driver waited for the sanitizer's shared GPU lock, then the three NCU
runs executed sequentially. No benchmark warmups/captures overlapped them.
                    original    late2       late3
initial regs/thread       94       80          56
blocks/SM (limits)         2        2           3
achieved occupancy    29.02%   35.60%      47.12%
issued warp/scheduler  0.56     0.59        0.62
warp cycles/issued     8.24     9.57       12.13

This confirms the intended occupancy increase without a speed win. NCU durations
are not used as graph timing measurements. Close this register-pool family rather
than tuning it further without a distinct reduction in work or dependencies.
