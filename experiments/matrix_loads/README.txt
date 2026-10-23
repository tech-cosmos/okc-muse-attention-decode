Session 9: native shared-matrix instruction reduction
===================================================

Experimental only. No production import or dispatch. Based on session-3
pair_mbar.py (copied as baseline.py). Production submission.py remains byte-for-byte
the 5e2441f/effc2bf source, SHA256 a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1.
Main468a53d has been merged; its new files are notes/profiles only. The add/add
profile conflict retained main's complete report rather than the locally extracted
kernel-only report. No Triton code or plans were changed.

Mechanisms
----------
* x4: one ldmatrix.x4.trans replaces two ldmatrix.x2.trans operations per PV
  fragment. Lanes 0-15 supply the first output-column group's row addresses;
  lanes 16-31 supply the second group's. This halves the matrix-load instructions.
* firstwrite: use output-only register constraints and scale-d=false for the
  first QK WGMMA, removing explicit score initialization. This is a NO-OP in
  generated stage4/page1 SASS: the baseline compiler already emits that operation
  with RZ / !UPT. baseline.sass and firstwrite.sass are byte-identical.
* vector: each consumer thread converts eight consecutive FP8 V bytes at once,
  using an aligned 64-bit shared load, four exact packed FP8-to-FP16 conversions,
  and a uint4 shared store, instead of four narrow loads/stores. Two iterations
  cover the same tile, replacing the previous eight iterations.
* combined = x4 + firstwrite; all = combined + vector.
* forced = x4 + vector with explicit st.shared.v4.u32 stores. nvcc/ptxas had
  scalarized one of vector's two uint4 stores; inline PTX keeps both at 128 bits.
  This variant does not include the no-op firstwrite source rewrite.

All variants retain every key/value, the same FP8 cache and FP8 query rounding,
FP16 probability/value matrix operands, FP32 partials, causal insertion of new
tokens, and all input-derived work inside each call. The shared destinations are
16-byte aligned: rows have 72 half elements and column offsets are multiples of
8. The gather and mbarrier synchronization topology is unchanged.

NVIDIA PTX references (consulted during this cycle):
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instruction-wgmma-mma
The documented matrix/address mapping supports x4; scale-d=false performs AB
without consuming the previous accumulator. No lower-precision approximation
was introduced.

Measurements
------------
H100 cold-L2 CUDA graph medians; empty replay subtracted; same-process comparison
under /tmp/okc-gpu.lock, including captures and reference checks. 60 rounds rotate
and reverse execution order. This is not the judge's balanced adjacent-pair
estimator: rotation occurs each round, including reversed rounds. Large gaps are
clear, but do not interpret sub-percent changes as established wins. JSON includes
raw per-variant times, empty median, and within-round deltas from native baseline.
Three separate runs; do not mix their absolute numbers to infer improvements.

                        verify B1       verify B16      verify B32
compare.json:
  production             18.320          155.040         303.872 us
  original native        34.688          229.840         439.376
  x4                     34.144          229.040         438.288
  firstwrite             34.736          229.552         440.224
  combined               34.272          229.360         438.896
compare-vector.json:
  production             18.512          155.616         304.928
  original native        34.528          231.056         443.392
  vector                 34.400          219.776         418.784
  all                    34.640          218.448         415.504
compare-forced.json:
  production             18.224          154.816         305.120
  original native        34.640          230.432         442.832
  all                    34.912          218.208         415.600
  forced                 34.944          210.976         402.000

The best native variant is 8-9% faster than the previous native at large batches,
but still 32-36% slower than production. No judge check or score is justified.
No arbitrary stage/register/grid sweep was done; every variant uses stages4 and
the same native split heuristic as the baseline.

Static SASS, stage4/page1 (whole function; pertinent operations occur once in
the cached-tile consumer loop):
                     baseline       x4          all             forced
  V LDS              8 x U16        same        2 x 64-bit       2 x 64-bit
  V STS              8 x 32-bit     same        4 x 32 + 1x128   2 x 128-bit
  V conversions      8              8           8                8
  LDSM               16 x x2        8 x x4      8 x x4           8 x x4
Resources remain 94 registers/thread and 62,528 shared bytes, no spills, two
resident CTAs per SM. Explicit store vectorization improves runtime without
increasing occupancy or reducing conversion count. Whole-function static counts
are archived in sass-counts.json; these are not dynamic instruction counts.

Reproduce from repo root with /home/daytona/okc/bin/python:
  experiments/matrix_loads/generate.py
  experiments/matrix_loads/compare.py
  experiments/matrix_loads/compare_vector.py
  experiments/matrix_loads/compare_forced.py
  experiments/matrix_loads/validate.py
  experiments/matrix_loads/finish_validation.py
  experiments/matrix_loads/sass.py
validate/fresh/bench_all accept VARIANT=all or forced, default forced. The all
full-reference suite was run before finish_validation.py, which runs the forced
suite, both full benches and fresh replays, then sanitizers and profiling in
strict sequence. No local GPU jobs overlap. Build caches are ignored.

Validation completed successfully:
* 39 variant/shape entries across the three comparison scripts (including repeated
  controls), with independent reference checks before and after graph capture.
* all and forced each pass 16/16 full tests: five native cases, eleven production
  fallback cases each. Full native/fallback benches after tests: 1.635x / 1.645x
  respectively, versus fixed SGLang/H200 reference numbers. These are rejected
  local experimental aggregates, not production regressions or judge results.
* 84 all-native fresh graph replays, 42 per variant across seven nearby shapes;
  TP8/DP, pages1/64, scales1/1/64, contexts10-120K, soft/coherent/edge inputs,
  changed QKV/cache/routes/lengths/positions/cache slots, including empty contexts.
* Isolated native Compute Sanitizer: memcheck 0 errors, synccheck 0 errors,
  racecheck 0 hazards/errors/warnings. Each tool runs 24 native calls across both
  variants, pages1/64 and scales1/1/64. Torch independently reduces the FP32
  partials, applies the output transform, and checks output/cache against reference.
* Runtime validation is stage4 only. Other compiled stages have no new runtime
  validation claim. Generation reproduces all six experimental source files.

All comparison/test/fresh/sanitizer processes exited before profiling. Isolated
NCU baseline/all/forced occupancy is 29.01/29.70/29.37%, issued warps per scheduler
cycle 0.56/0.53/0.55, warp cycles per issued instruction 8.22/8.91/8.56. Resources
stay unchanged. Do not use profiler durations as graph measurements or interpret
these counters as dynamic counts of a particular instruction. NCU warns that the
optional PC sampling metric is absent; no detailed stall attribution is claimed.

Outcome: a useful native staging improvement, but no production winner. Keep
forced.py as the better native building block if a distinct future architecture
is justified. Do not repeat tiny stage/register/load-width sweeps of this family.
Current submitted source is still pending as a48b8c849e53426b. This cycle used
zero checks and zero scored runs; detailed budget/queue handoff is NOTES-codex.md.
