Session 20: native sliding FP32 partial-store vectorization
=========================================================

This is an isolated experiment on codex-exp/cuda-sliding-stores. Production
submission.py stays byte-identical to 45c56bc. No Triton/Gluon kernel, plan,
merge, attention arithmetic, precision, or dispatch is changed.

The single-tile native epilogue writes adjacent FP32 accumulator values with
16 scalar stores, plus two scalar max/sum stores. Session19's instruction
profile shows excess theoretical global sectors at these sites, but few stall
samples. This motivates a bounded test, not a claim that stores dominate.

base.cu is extracted exactly from production. generate.py creates:
  vector: C++ float2 assignments for adjacent values and max/sum.
  explicit: volatile inline PTX st.global.v2.f32 for those same pairs.
Every address is eight-byte aligned by the existing FP32 partial layout.
All asynchronous waits, fences, and cache writes remain unchanged.

SASS page1/EARLY shows that nvcc scalarizes the C++ accumulator stores:
  base:     18 scalar partial stores
  vector:   16 scalar + one 64-bit partial store
  explicit: nine 64-bit partial stores
All have 74 registers, 17408 shared bytes reported by cuobjdump (16384 ptxas),
zero stack/local allocation and no spills. Matrix and CTA-barrier sites stay
unchanged. results/audit.json defines the static instruction-count convention;
do not compare those counts with earlier reports using different NOP handling.

Reproduce from repo root using /home/daytona/okc/bin/python:
  python experiments/sliding_stores/generate.py
  sh experiments/sliding_stores/run.sh

run.sh runs 16-case reference suites, full benches, paired graph timings and
fresh graph replays for each candidate. The broad experimental dispatcher
forces native on eligible L1 sliding shapes, including B16/32 where production
uses Triton; full-bench aggregates are not production performance claims.
Custom drivers lock generation, reference checks, captures and all replays.
bench.py uses the standard harness timing-loop lock. Our work ran sequentially
through those full benches; no profiler or sanitizer overlapped their lifetime.

Additional commands (export STORE_VARIANT=explicit and TORCH_EXTENSIONS_DIR
as in run.sh):
  python experiments/sliding_stores/compare.py --seed 711 --rounds 120 --suffix=-confirm
  python experiments/sliding_stores/compare.py --seed 89 --rounds 120 --suffix=-fixed --fixed-one
  compute-sanitizer --tool memcheck --error-exitcode 99 python experiments/sliding_stores/sanitize.py
  compute-sanitizer --tool synccheck --error-exitcode 99 python experiments/sliding_stores/sanitize.py
  compute-sanitizer --tool racecheck --error-exitcode 99 python experiments/sliding_stores/sanitize.py
  ncu --kernel-name regex:sliding --launch-skip 2 --launch-count 1 --section SourceCounters --warp-sampling-interval 0 --import-source no --page source --csv --print-source sass python experiments/sliding_stores/profile.py base early
  # Repeat the profile command with explicit instead of base.
  python experiments/sliding_stores/profile_summary.py
  python experiments/sliding_stores/audit.py

Graph comparisons alternate adjacent AB/BA order, flush 512MB before every
replay, and subtract empty replay cost. Independent output/cache reference
checks run before capture, after initial replay and after timing. Raw pairs
are archived. The median paired delta can differ from the difference between
the two independently computed medians. Initial/confirmation B1 uses agentic
lengths; the separate fixed-one run includes the exact scored B1 shape.

Each variant's fresh.py runs 60 all-native graph replays, changing QKV, caches,
routes, lengths, positions and cache slots across TP8/DP, pages1/64, two scales,
short/long/empty contexts and soft/coherent/edge distributions. The explicit
sanitizer driver runs 12 native calls per tool and independently reduces FP32
partials with torch before output/cache checks. Runtime coverage is EARLY=True;
EARLY=False is compiled only. No separate vector sanitizer coverage is claimed.

Setup note: the first production import was stopped before GPU work to avoid
an existing default extension-cache lock. Builds then used an isolated repo
TORCH_EXTENSIONS_DIR. The old shell's delayed "Terminated" message appears in
the otherwise complete vector test log; it is not a kernel/test failure.
Generated compiler/SASS report trailing whitespace is stripped for git hygiene.

Result: not promoted. The explicit variant's B1 agentic paired changes are
-0.096us initially and -0.048us at the second seed; B3/B8 regress slightly.
Fixed B1 ctx40K/80K/120K paired changes are -0.048/-0.096/0.000us. Small,
uneven end-to-end changes do not justify a new dispatch or H200 check.
NCU confirms partial-store executed warp instructions1152->576, theoretical
L2 sectors8448->4224, excessive theoretical sectors4224->0. These are source
metrics, not measured HBM bytes. Total executed warp instructions only fall
164482->163394; shared excess wavefronts stay4104. Profiler durations are not
graph times. Baseline and candidate profiles take the same shared GPU lock as
the fixed-context comparison, so their GPU operations do not overlap.

Full result and judge status: NOTES-codex.md session20 and archived audit/profile
summaries. No experiment is imported by production. Production remains unchanged.
