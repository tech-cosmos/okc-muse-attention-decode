Session 6: native FP8 V operands without FP16 shared staging (rejected)

baseline.py is the previous native paired-head/mbarrier prototype copied from
codex-exp/cuda-gather:experiments/gather/pair_mbar.py. make_direct.py generates
direct.py by removing its shared FP16 V array, per-tile cooperative conversion,
and one consumer barrier. ldmatrix.trans loads FP8 column pairs as b16 words;
warp shuffles select the column pair and byte permutations extract each column
across adjacent K rows. Exact FP8-to-FP16 conversions feed the unchanged PV MMA.
All attention/cache arithmetic, every key, FP32 partials, scales and masks remain.
No Triton source or plans were tuned. This experiment is never imported by
submission.py and was never submitted.

Primary reference for the fragment mapping:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float

Reproduce from root with ~/okc activated:
  python experiments/direct_v/make_direct.py
  python experiments/direct_v/bench_direct.py
  python experiments/direct_v/validate_direct.py
  python experiments/direct_v/fresh.py
  python experiments/direct_v/bench_all.py
  /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 7 python experiments/direct_v/sanitize.py
  (likewise --tool synccheck and --tool racecheck)

H100, cold-L2 CUDA graphs, median minus empty replay, shared GPU timing lock:
  batch, verify/full  old native S4   direct V S4   production
   1, 80K             34.576 us      35.152 us      18.400 us
  16, 80K            231.152 us     239.872 us     155.072 us
  32, 80K            446.672 us     454.080 us     301.552 us
S3/S5 did not rescue the path (see results/bench.json). These are same-process
measurements, not a balanced paired-run speedup estimator. The huge production
gap is decisive. Full experimental native/fallback bench: 1.585x versus the
fixed SGLang/H200 reference; not a judge result or production regression.

Why it loses: the S4/P1 code has 96 registers vs 94 and 53,312 shared bytes vs
62,528; still limited to two CTAs/SM. The changed loop has 8 instead of 16 LDSM,
removes 8 LDS.U16, 8 STS, and one dynamically executed consumer barrier, but adds
32 SHFL.IDX, 32 PRMT, and raises FP8->FP16 unpack count from 8 to 32. Duplicate
conversion per consumer warp costs more than staging the shared V tile once.
results/*.sass contain only S4/P1 cuobjdump functions; sass-counts.json counts
whole functions (two static branch-specific BAR instructions disappear).
Do not interpret these static counts as Nsight dynamic instruction counters.

Validation:
- 15 independent attention reference comparisons (five shapes x stages 3/4/5).
- Full 16-case native/fallback reference suite passes; five cases invoke native.
- 42 all-native fresh graph replays on seven nearby shapes, pages 1/64, TP8/DP,
  scales 1 and 1/64, soft/coherent/edge data, contexts 10-120K. Copy fresh tensor
  values and routing, change lengths/positions/cache locations on device, include
  empty contexts; all independently reference-checked.
- Isolated native attention: memcheck and synccheck 0 errors; racecheck 0 hazards,
  0 errors, 0 warnings. Four small page/scale shapes, three changing-QKV calls each.
  This intentionally excludes Triton's unrelated driver-attribute probing. The
  sanitizer driver checks finite partials; full output correctness is covered by
  the independent reference suite/replays above.

No winning implementation, production dispatch, new judge check, or scored job.
Close this direct-register variant; do not repeat without reducing its added
shuffle/permutation/conversion cost.
