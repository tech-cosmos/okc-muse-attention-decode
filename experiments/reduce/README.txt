Codex session 4: native hierarchical full-layer reduction
========================================================

The production candidate adds eight native CUDA warps per output query head.
Each warp merges up to 16 FP32 splits using a local maximum and denominator;
a shared-memory reduction combines those eight partials. All normalization,
gating and input reads stay in kernel_fn. Coherent volatile PTX loads follow
griddepcontrol.wait; no input-derived state persists between calls.

Dispatch: full layers, batch * num_kv_heads <= 4, 64 < actual splits <= 128.
Triton attention plans are unchanged. During this session main 651b421 landed
with a faster sliding merge. It beat the previous native sliding path by
~0.3-0.4 us in paired tests, so the candidate removes that old native path.

Source and reproducibility
--------------------------
Activate /home/daytona/okc before running. Work from the repo root. All GPU
correctness/stress/profiling runs acquire /tmp/okc-gpu.lock; time_graph takes
that lock for its replay loop and uses cold L2 and empty-replay subtraction.

native.py: generic 1/2/4/8-warp reduction experiment. Native wrappers replace
only submission._combine, leaving attention and plans untouched.
native_v1.py: initial 32-split-minimum version. Do NOT use its 1-warp variant
for >128 splits: the initial sweep caught that unsupported capacity. The
later version asserts capacity and the sweep skips unsupported configurations.

sweep.py: basic generic warp-count comparison.
tiled.py / compare_tiled.py: split/dimension tiles (8 or 16 lanes per row),
  vectorized 16-byte loads. Did not win.
cluster.py / compare_cluster.py: distribute a head over CUDA cluster CTAs,
  shared-memory DSM combination with barriers before and after remote reads.
  Did not win. Scored comparisons only; no sanitizer/full-suite claims.
specialized.py / compare_specialized.py: compile-time head/token indexing.
  Shorter instruction prologue did not improve end-to-end times; PDL already
  overlaps indexing with attention. Scored comparisons only.
paired.py: alternating-order 5-pair sweep across scored and nearby batches.
final_compare.py: integrated candidate vs exact prior Triton full merge,
  14 scored/nearby page/context/batch combinations, five paired repeats.
sliding_main_compare.py: compares the OLD native sliding path to main 651b421.
  Run on the merge commit before removal of that path; current production
  intentionally no longer exports the old native combine function.

Historical experiment scripts replace _combine. Once native full dispatch is
integrated, the small-batch full calls bypass those wrappers; use these earlier
scripts with the pre-integration parent 5a21d0f (plus experiment files), or with
that new native dispatch disabled. final_compare.py works on the final source.
integrate.py is a historical source-writing script for the pre-integration
source only; do not rerun it on the final candidate.

Validation
----------
python tools/bench.py test
python tools/bench.py bench
python experiments/reduce/validate.py fresh production 8
python experiments/reduce/stress_final.py

The original generic hierarchy passed 16/16 and 72 fresh-value/routing/length
graph replays. stress_partials.py passed 162 independent torch comparisons
with changing split weights, partial vectors, gates and scales. The integrated
candidate also passed 16/16 and 72 fresh graph replays before main's merge.
Final production validation files have the final- prefix.

Generic native synccheck: zero errors. Generic native racecheck: zero hazards.
The integrated memcheck run reported two CUDA_ERROR_INVALID_VALUE API calls
from Triton's loadBinary. driver.c lines 313-328 attempts the newer oversized
shared-memory attribute, then deliberately falls back when unsupported. No
invalid memory accesses were reported; do not call that whole log zero-error.
Final-native-* sanitizer logs isolate the actual production native merge and
avoid invoking Triton's loader. They preserve full memory/race/sync checking.

Measurement caveats
-------------------
All numbers here are local H100 timings, not H200 judge results. Low-single-us
changes on large full-attention batches were inconsistent; dispatch therefore
stays narrow. Early sweep-v1-incomplete.log ends at an unsupported 256-split,
one-warp invocation. Its earlier supported measurements are valid; the failed
invocation is not a candidate. No experimental variant was submitted.

Final measurements and validation
---------------------------------
Candidate fddd5c1 includes main 651b421. Full local suite 16/16; full local
bench 1.882x vs the fixed SGLang-on-H200 reference. final-confirm.json holds a
last sequential paired confirmation and short B2 decode coverage. These are
not H200 judge timings. final-native-memcheck/synccheck each report zero errors;
final-native-racecheck reports zero hazards, errors and warnings. Each isolated
sanitizer run checks 42 fresh graph replays against torch over seven shapes.
final-fresh.log holds 72 fresh-input/routing/length graph checks of the final
integrated production dispatch and inherited fallbacks.

merge-profile-details.txt: standalone non-PDL merge, B1/L8/S126. Native 48 regs,
2176 shared bytes, 11.79% achieved occupancy; Triton 80 regs, 1024 dynamic shared
bytes, 6.22% achieved occupancy. Cold standalone kernel times 5.50 vs 5.57 us.
These profiler times are NOT the PDL-overlapped kernel_fn comparison; partials
are cold here, unlike their production predecessor relationship.
final-native.sass confirms ACQBULK precedes all coherent partial reads in the
PDL specialization. The two earlier ordinary reads load gate/scales only.

cooperative.py / compare_cooperative.py: late exploratory alternative with
warp-zero split weights cooperatively computed and shuffled. Verify times tied;
one decode comparison showed ~0.4 us saved, but this small result is unconfirmed.
It is not integrated, not submitted, and only has six baseline comparisons
(three also have independent reference checks). Retain as a possible next-cycle
idea, not an established improvement or a full-suite/sanitizer-validated variant.

No Triton attention plans were changed by Codex. Historical scripts' old
wrapper path and removed native-sliding binding require their matching source
commit; final scripts target production fddd5c1. Judge job/result and promotion
status are recorded in NOTES-codex.md and the shared coordination board.

Later main merge (65ff1ab)
--------------------------
Production source effc2bf merges main's new-token-before-cache change. Native
code and dispatch are unchanged. main65-test.log: 16/16. main65-fresh.log: 72
fresh routing/value/length replays. main65-bench.log: 1.909x. main65-confirm.json
is the same-process comparison against the latest inherited Triton full merge.
The final-* prefix above denotes the earlier fddd5c1 revision, not this later
main merge; both revisions are explicitly recorded in NOTES-codex.md.
Check 40e1aa41415a4c87 covers fddd5c1. Check 01bc4ecc9c7047d9 covers effc2bf.
Both were queued at 22:28 UTC. Neither is a scored run. Do not resubmit either
check, and do not score effc2bf using the result of the earlier source revision.
