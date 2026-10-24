Session 26: integrated main validation and native full-merge neighborhood
2026-10-24. Offline complete-output comparisons; no experimental kernel.

Production merges main a794ac6 as 8c35564. SHA256:
6f0fcb357146bbfd4e69e3f9d8a3ea2d9968f09c26b5876a0aa83af63ebe8c05
The only changed top-level function versus previous production is main's _plan.
Native extensions and kernel_fn dispatch retain exact ASTs. Shared helpers match
main. Nothing under experiments is imported by production.

Frozen sources:
- main_snapshot.py: exact origin/main a794ac6 (Triton/Gluon, no native additions).
- previous_snapshot.py: exact f2f34b4 production, SHA6b171beb8ed5.

The integration suite compares previous_snapshot (JSON main_us) against current
production (codex_us), across 13 full-verify shapes: B4/5 controls, B6/8/12/16/17/
24/32/40/64 at80K, DP2/page64 at20K, B6/page64/mixed at4K. Fresh integration
replays vary all tensors across four inputs each, shortening lengths to stay
inside each generated request's assigned page-table range; final input is empty.
The two sources each get independent output/cache reference checks.

The neighborhood suite compares main_snapshot against current production across
44 full L8 shapes. B1/fixed and B2/agentic at7K,8192,9K,10K,12K,16K,20K,40K on
pages1/64; additional B2/fixed at8192,9K,10K,12K,16K,20K on both pages. The only
implementation difference exercised here is the existing native small-full merge.
No shape-specific dispatch threshold or native architecture is fitted to timings.

Both suites use complete kernel_fn output, 80 adjacent AB/BA graph pairs per
shape,512MB L2 flush before each timed replay, and empty-replay subtraction.
Shared GPU lock covers input generation, reference checks, capture and all replays.
Each implementation is reference-checked before capture, after first graph replay,
and after timing. Each raw triplet is baseline/current/empty in microseconds.
Paired deltas need not equal differences of separately computed medians.
Block-bootstrap intervals quantify within-run variation, not cross-run/H200
uncertainty or multiple-comparison correction. Do not infer total GPU time by
summing PDL profiler rows. This cycle makes no new sanitizer claim.

Reproduce from repository root, in ~/okc environment, sequentially:
python tools/bench.py test
python tools/bench.py bench
python experiments/native_neighborhood/compare.py compare --suite integration --rounds 80 --name integration
python experiments/native_neighborhood/compare.py fresh --suite integration
python experiments/native_neighborhood/compare.py compare --rounds 80 --name neighborhood
python experiments/native_neighborhood/audit.py

Bench.py retains its existing timing-loop lock. The custom drivers also lock
setup and validation. No concurrent local profiling, sanitizer or benchmark work.
The archived production standard-suite and bench logs live at
profiles/session26-integrated-{test,bench}.log on codex.

Integrated results (seed25103, H100 microseconds):
 batch/shape       previous/current         median paired delta
 6/80K TP8         68.432 / 69.840          +1.312
 8/80K TP8         85.200 / 83.280          -2.176
12/80K TP8        122.384 /115.968          -6.512
16/80K TP8        153.264 /145.296          -7.824
17/80K TP8        160.720 /155.664          -5.040
24/80K TP8        211.856 /204.928          -7.216
32/80K TP8        273.408 /268.208          -4.912
40/80K TP8        330.464 /330.304          -0.096
64/80K TP8        515.312 /512.016          -3.792
 2/20K DP          51.936 / 49.216          -2.736
 6/4K mixed        17.136 / 17.216          +0.112

B4/B5 are unchanged controls (-0.336/+0.048us). B40 has unchanged split count
and source path; its small delta is variation. The B6 regression is recorded for
Claude, who owns Gluon dispatch/plans. Codex made no plan adjustment. Do not
attribute the integrated improvement to native kernels; those did not change.

Neighborhood initial scan: all20 native page1 cases favor native (-1.136 to
-0.272us); two page1 fallback controls are flat. Page64 B1 native cases all win
(-0.896 to-0.576us). B2/page64 is flat or slower around small S: agentic8192/S69
+0.032us,9K/S76 0us,16K/S68 +0.320us, fixed8192/S65 0us. Other native B2/page64
cases win, up to-1.872us at40K. S is the actual split count, not monotonic with
context: chunk-size transitions can reduce it (e.g. agentic16K/S68). This scan
supports a small-split B2/page64 weakness but no shape-general cutoff; the same
small split counts win on page1 and B1. No native dispatch change is made.

Second-seed confirmations (25104) keep selected suspect/neighbor/control entries:
python experiments/native_neighborhood/compare.py compare --indices 3 5 7 25 27 29 38 39 40 --seed 25104 --rounds 120 --name neighborhood-confirmation
python experiments/native_neighborhood/compare.py compare --suite integration --indices 1 2 3 --seed 25104 --rounds 120 --name integration-confirmation
These are additional complete-output graph checks, not independent broad scans.

Confirmation results: B2/page64 agentic8192/S69 +0.128us,9K/S76 +0.144us,
10K/S85 -1.632us; corresponding page1 cases -0.832/-1.248/-0.368us. Fixed
B2/page64 at8192/S65 +0.032us,9K/S71 -0.272us,10K/S79 -0.960us. The narrow
weakness remains small (~1-3% in these cases); no broad fallback rule is promoted.
The16K/S68 +0.320us initial observation has no second-seed confirmation this cycle.
Do not claim native is always faster on nearby shapes.

Integration confirmation: unchanged B5 control +0.080us, B6 -1.056us, B8
-4.240us. The original B6 +1.312us reverses with fresh inputs; neither result
supports a universal B6 regression/gain. Both are retained for Claude's review.
No favorable-seed selection or change to the inherited plan occurred.

Totals: 57 distinct measured shapes,12 confirmation entries,6000 adjacent pairs,
414 independent output/cache reference checks in timing drivers;104 fresh graph
replays (52 per implementation). Of current-source fresh replays,44 use Gluon
and8 use unchanged Triton fallback. Previous-source replays are separate controls.
Integrated standard suite16/16, full bench1.965x vs fixed SGLang/H200 numbers.
This aggregate is a local benchmark, not a new H200 estimate or a paired score.
Audit validates exact source hashes,14 shared main helper ASTs, unchanged native
extension/dispatcher ASTs and complete result/log counts. Native kernels retain
prior sanitizer coverage; no new sanitizer run was necessary for a plan merge.
