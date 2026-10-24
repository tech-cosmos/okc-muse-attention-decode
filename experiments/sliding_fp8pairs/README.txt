Session 21: paired FP32-to-FP8 conversions in native sliding attention
===================================================================

This isolated experiment belongs to codex-exp/cuda-sliding-fp8pairs. It is
not imported by production. Production includes main0a22b94's producer-side
V conversion, integrated as9064c53; its native kernels are unchanged.

Motivation: session20 SourceCounters/SASS show20 scalar FP32-to-E4M3 sites
using packed instructions with an unused zero operand. Pairing adjacent Q
values can reduce conversions and byte packing without changing arithmetic,
precision, cache format, query rounding, synchronization or all-key attention.
This is an instruction-reduction test, not another load-placement sweep.

generate.py starts from base.cu, extracted exactly from production's embedded
CUDA, and makes two variants:
  query: pair the two adjacent RoPE query values per conversion.
  both: additionally pair new K and V conversions (split the resulting bytes
        back into the existing lane and lane+32 store locations).
The CUDA13 header intrinsic uses cvt.rn.satfinite.e4m3x2.f32, the same rounding
and saturation as scalar conversions. No half-precision intermediate is added.

Page1/EARLY SASS, base/query/both:
  FP32-to-FP8 conversion sites20/12/10; all F2FP sites52/44/42;
  PRMT sites8/0/0; LOP3 sites49/29/31.
All retain74 registers,17408 shared bytes from cuobjdump (16384 ptxas),
zero stack/local allocation/spills,2QGMMA/4HGMMA and3CTA barrier sites.
These are static instruction sites; SourceCounters results are separately
recorded in results/profile-summary.json. Profiler duration is not graph timing.

Result: rejected. Balanced adjacent AB/BA cold-L2 graph pairs,60 rounds per
shape, independently reference-checked before capture, after replay and after
timing, use the shared GPU lock throughout and subtract an empty replay.
Both target the exact scored fixed-length B1ctx80K/page1 shape:
  query: parent4.336/candidate4.352us, paired delta+0.016us;
  both: parent4.304/candidate4.320us, paired delta-0.016us.
B8 paired deltas+0.112/+0.096us; nearby shapes are mixed. No H200 check or
production dispatch is justified for these native changes. Per-variant medians
need not subtract to the median paired delta. Each variant has8shape comparisons.

Validation:
  Both16/16 full-reference tests (3native,13production fallback), then full
  broad-native benches1.911/1.910x against fixed SGLang/H200 numbers. Those
  aggregates are experimental, not production regressions or H200 estimates.
  Each60all-native fresh graph replays across10shapes: TP8/DP, pages1/64,
  scales1/1/64, contexts10-120K, soft/coherent/edge inputs, changed QKV/cache/
  routes/lengths/positions/cache slots and empty contexts.
  bitexact.py compares FP32 partials and new cache bytes with native baseline
  on12case/seed combinations per variant (24comparisons total).
  Both-conversion variant alone has memcheck/synccheck/racecheck coverage;
  each tool runs12native calls and independently reduces partials using torch
  before output/cache reference checks. See complete sanitizer summaries.
  Runtime coverage is EARLY=True; EARLY=False is compiled only.

Reproduce from repository root:
  python experiments/sliding_fp8pairs/generate.py
  sh experiments/sliding_fp8pairs/run.sh
  /home/daytona/okc/bin/python experiments/sliding_fp8pairs/bitexact.py
  PAIR_VARIANT=both /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 99 /home/daytona/okc/bin/python experiments/sliding_fp8pairs/sanitize.py
  # Repeat with synccheck and racecheck.
  /usr/local/cuda-13.0/bin/ncu --kernel-name regex:sliding --launch-skip 2 --launch-count 1 --section SourceCounters --warp-sampling-interval 0 --import-source no --page source --csv --print-source sass /home/daytona/okc/bin/python experiments/sliding_fp8pairs/profile.py base early
  # Repeat with query and both; save results/profile-{variant}.csv and .log.
  python experiments/sliding_fp8pairs/profile_summary.py
  python experiments/sliding_fp8pairs/audit.py

The production integration is separately tested and timed by integrated_fresh.py
and integrated_pairs.py; production_before.py is exact45c56bc, the source of
the older pending checkb4853b2accdc4b50. These drivers validate main's merge,
not the rejected conversion experiment. Integrated test/bench/fresh/pair logs
are archived under profiles/session21-* on codex. See NOTES-codex.md session21
for production hashes, judge state/budget and final handoff.
