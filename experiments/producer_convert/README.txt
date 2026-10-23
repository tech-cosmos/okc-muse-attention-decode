Session 7: move FP8 value conversion off the attention consumer path
==================================================================

No production change or native attention dispatch. Both mechanisms lose against
both the existing paired-head native prototype and production Triton attention.
No experimental source was sent to the judge. Full native/fallback graph bench
was 1.592x (direct4) / 1.515x (async2) against fixed SGLang/H200 times; these
are rejected local experiments, not production regressions.

Native sources (all full verify, L=8, G=8, D=64, even Hkv, page1/64):
- baseline.py: exact session-3 pair_mbar.py, copied for reproducibility.
- producer.py: producers cp.async K, load aligned uint4 V directly, convert each
  FP8 pair exactly to FP16, and store staged shared V. Separate mbarriers publish
  async K and generic V stores. Consumers wait on both, use the same FP8 WGMMA QK
  and FP16 MMA PV, but no longer convert V or synchronize after conversion.
- async_convert.py: preserve asynchronous K+V gathers in two producer warps;
  use two/four dedicated conversion warps between gather and attention. Converted
  V uses a separate shared ring. Consumers still compute every key/value and
  final partials remain FP32. Nothing is cached across calls.
- async_two.py: first two-converter source, for the original compare-async logs.
- build_variant.py/build_async.py deterministically regenerate the current two
  variants from baseline.py. Original build-async.log belongs to async_two.py;
  build-async-v2.log belongs to current async_convert.py.

Measurements
------------
All end-to-end numbers are H100 cold-L2 CUDA graph replay medians with the shared
GPU timing lock, empty replay subtraction and 60 rounds. Each measured variant
was checked independently against reference.py before timing. These sequential
same-process comparisons are not a balanced paired estimate. Gaps are large.

Direct producer run, stages=4:
                           B1 verify     B16 verify     B32 verify
  production                18.480         156.544        306.368 us
  original native           34.736         230.672        442.080
  two direct producers      47.104         396.368        773.024
  four direct producers     36.192         241.488        467.808

Asynchronous converter run, stages=4:
                           B1 verify     B16 verify     B32 verify
  production                18.400         154.704        304.432 us
  original native           34.736         230.640        442.208
  two converters            39.648         269.856        522.752
  four converters           45.648         363.408        719.824

Direct producers swept 2/4 warps and stages 2/3/4/5. Async conversion swept
stages 2/3/4 at two warps, then stages 3/4 at two/four warps. None approaches
production, so no split-count sweep or judge check is warranted.

Resource facts from ptxas, page1/stages4:
- Original native: 94 registers/thread, 62,528 shared bytes, 320 threads.
- Direct, two producers: 89 registers, 69,728 shared bytes, 320 threads, no spills.
- Direct, four producers: 80 registers, same shared memory, 384 threads, no spills.
- Async, two converters: 80 registers, 90,208 shared bytes, 384 threads, no spills.
- Async, four converters: 72 registers, same shared memory, 448 threads, spills.
  Consult build-async-v2.log for exact spill counts by template specialization.

The direct design moves V HBM latency onto synchronous producer instructions.
The pronounced improvement from two to four producer warps is consistent with
that bottleneck, but does not prove its entire cause. Async conversion avoids
those loads but introduces another stage dependency and limits conversion to
fewer warps. Extra converters require a tighter register cap and spill. These
changes remove consumer instructions without reducing total per-tile work enough.

Validation
----------
- 45 native scored-shape reference comparisons across the three sweeps (24
  direct, 9 original async, 12 current async; some configurations repeat).
- Retained direct4/stage4 and async2/stage4 each pass 16/16 full reference tests:
  five native full-verify cases, eleven production fallback cases.
- Each retained variant passes 42 all-native fresh graph replays across seven
  nearby shapes: pages1/64, TP8/DP, scale1 and 1/64, ctx10..120K, soft/coherent/
  edge data, new cache/QKV/page routes and changed lengths/positions/cache slots,
  including empty contexts. 84 replays total.
- sanitize.py isolates only native launches (no Triton module loading) and
  independently reduces their FP32 partials in torch before reference checking.
  For each sanitizer, eight variant/page/scale combinations with three changed
  QKV calls each = 24 native reference checks. Final sanitizer results are in
  memcheck.log, synccheck.log, racecheck.log; run_sanitizers.py fails on errors.
  Both retained kernels: memcheck 0 errors, synccheck 0 errors, racecheck 0
  hazards/errors/warnings.
- bench-all-*.log use retained native dispatch plus production fallback for all
  12 scored shapes, after the reference suites. They describe experiments, not
  a production change. Reproduce with VARIANT=producer or VARIANT=async.

Reproduction from repo root (activate ~/okc first):
  python experiments/producer_convert/compare.py
  python experiments/producer_convert/compare_async.py
  python experiments/producer_convert/compare_async_v2.py
  VARIANT=producer python experiments/producer_convert/validate.py
  VARIANT=async python experiments/producer_convert/validate.py
  VARIANT=producer python experiments/producer_convert/fresh.py
  VARIANT=async python experiments/producer_convert/fresh.py
  python experiments/producer_convert/run_sanitizers.py
  VARIANT=producer python experiments/producer_convert/bench_all.py
  VARIANT=async python experiments/producer_convert/bench_all.py
  python experiments/producer_convert/run_profiles.py

Do not run profile collection concurrently with timing/capture activity. The
runner takes the shared GPU lock, but the inherited bench helper warms and
captures graphs before taking its timing lock. NCU kernel durations are profiling
measurements, not the end-to-end CUDA graph numbers reported above.

The first NCU collection may have overlapped unlocked benchmark warmups; it is
retained under preliminary/ and excluded from conclusions. Final profiles were
recollected after every benchmark/test/sanitizer process completed. See its README.

Final isolated NCU counters (B16, stage4, one attention kernel; NOT graph timing):
                           original     direct4     async2
  registers/thread              94           80         80
  achieved occupancy (%)     29.00        35.76      33.88
  issued warp inst/cycle       0.56         0.55       0.56
  DRAM throughput (%)        35.96        38.70      30.83
  barrier stall (%)           9.16         5.50       6.12
  long scoreboard stall (%)  13.93        31.87      31.10
  short scoreboard stall (%) 16.19         8.64       9.75

Removing consumer conversion lowers barrier/short-scoreboard stalls and raises
occupancy, but does not increase instruction issue rate, while long-scoreboard
stalls rise. These whole-kernel counters do not identify the exact stalled load.
Use the cold graph numbers above for the performance decision, not NCU durations.
