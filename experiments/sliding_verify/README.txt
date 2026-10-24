Session14: native L8 sliding single-tile attention
================================================
Experimental only; production submission.py is unchanged.

Source families
---------------
production_parent.py: exact production3fb391f/72dfa26 source at session start.
generate.py -> kernel.cu/native.py: derived from the prior L1 direct-register-Q
kernel. Every warp supplies16 useful rows (two tokens, eight Q heads each), for
all64 WGMMA rows. Each CTA handles64 consecutive positions for one KV head.
The union of eight shifted2048-token windows has up to2055 positions, so the
launch needs up to33 partials. Per-row masks enforce its causal upper edge and
its individual lower window edge. Each new token is inserted into its natural
tile, from its own QKV projection, only after the historical zero-fill gathers
complete. A single owner writes each new cache row. No attention loads race with
those cache writes. Both EARLY and LATE orderings compute identical arithmetic.

vectorize.py -> vector.cu/vector.py: one explicit vector-load control for L8's
full query tile: 64-bit Q and cos/sin loads. The prior L1 vector experiment lost;
here all64 rows are useful, making query setup eight times larger. This is a
measured control, not a reason to revisit L1.

precompute.py -> precomputed.cu/precomputed.py: prepare Q operands and new K/V
once per request/head in a separate native kernel, inside every timed call.
Native attention uses dependent launch, issues independent historical gathers
before waiting for preparation in EARLY mode, then reads fresh Q/new-KV scratch.
The query reads use volatile coherent PTX loads after griddepcontrol.wait. Own
new-token copies start only after zero-fill copies finish. The inherited merge
uses PDL after attention. This is a three-kernel timed path; no derived values
are cached or prepared outside kernel_fn.

All retain FP8 E4M3 cache/queries, FP16 PV operands, FP32 partials, every key,
normalization, gates, cache scales, and BF16 output. No Triton plan/source tuning.

Reproduce in the documented ~/okc venv with CUDA_HOME=/usr/local/cuda-13.0:
  python experiments/sliding_verify/generate.py
  python experiments/sliding_verify/vectorize.py
  python experiments/sliding_verify/precompute.py
  python experiments/sliding_verify/compare.py
  python experiments/sliding_verify/compare_vector.py
  python experiments/sliding_verify/compare_precomputed.py
  python experiments/sliding_verify/validate.py
  python experiments/sliding_verify/validate_vector.py
  python experiments/sliding_verify/validate_precomputed.py
  python experiments/sliding_verify/fresh.py
  python experiments/sliding_verify/fresh_precomputed.py
  compute-sanitizer --tool <memcheck|synccheck|racecheck> --error-exitcode 99 \
    python experiments/sliding_verify/sanitize.py
  compute-sanitizer --tool <memcheck|synccheck|racecheck> --error-exitcode 99 \
    python experiments/sliding_verify/sanitize_precomputed.py

All custom GPU drivers hold /tmp/okc-gpu.lock throughout their GPU work. Full
bench drivers use tools/bench.py's existing timed-loop locking. Do not run a
full bench concurrently with other timing/profiling; its warmups are unlocked.
Compare scripts retain40 balanced adjacent AB/BA pairs per shape/order variant,
flush512MB before every replay, and subtract empty replay cost. Each variant is
checked against independent output/cache references before and after capture.
parts.py is explicitly an offline diagnostic omitting the merge only; its
attention-only outputs are not submitted or claimed to compute final outputs.

Results and validation coverage are recorded below after completion.

Measured outcome
----------------
No variant wins; none is promoted or sent to the judge. Independent balanced
cold-L2 graph comparisons, microseconds, EARLY variants (parent/native):

                    B1               B16               B32
native         5.784 / 7.952    12.632 / 28.544    20.200 / 53.096
vector         5.664 / 7.272    12.568 / 27.456    20.160 / 51.176
precomputed    5.760 / 8.256    12.704 / 29.728    20.080 / 52.160

The tables are separate comparison runs; compare each native variant with its
own paired parent. Small differences across runs are not isolated speedups.
All tested nearby shapes lose too, including B1 ctx10/page64. See the raw JSON
for B2, DP, page64,1/64 scale, edges, and coherent inputs and both orderings.

Offline parts.py diagnosis, B1 ctx80K:
  parent full/attention-only: 5.728 /4.112 us
  native LATE full/attention-only: 7.760 /5.952 us
  native EARLY full/attention-only: 8.016 /6.368 us
The merge is not the main explanation for the small-batch regression. Removing
repeated Q preparation also fails to improve the end-to-end call: the extra
launch, scratch accesses and dependency remain. This is not a claim that any
single one of these costs has been measured in isolation.

SASS and resources:
  native LATE/EARLY:79/74 registers, vector80/80, precomputed attention75/75,
  prepare42. All zero stack/local allocations and spills. Attention shared
  allocation17408 bytes in cuobjdump (ptxas16384); prepare0.
  Native page1 scalar load sites45 LDG.E +40 LDG.E.U16 +2 LDG.E.64;
  vector13 +8 +26, respectively. Two QGMMA and four HGMMA remain.
  Prepare-once attention758 static instructions vs native1457/1481, but it
  adds a separate preparation kernel. Attention has no RSQ and its Q loads
  occur after ACQBULK; EARLY historical gathers precede ACQBULK. Instruction
  counts are static SASS, not a claim about dynamic throughput.

Validation logs:
  All six family/order combinations pass16/16 full tests, three native L8
  sliding cases and13 production fallbacks each. Every suite is followed by
  a full native/fallback bench. Geomeans vs fixed SGLang/H200 numbers:
  native1.729/1.723, vector1.755/1.735, precomputed1.703/1.711 (LATE/EARLY).
  These are experimental local aggregates, not production or judge results.
  native and precomputed each have84 all-native fresh graph replays,42/order,
  seven shapes covering TP8/DP, page1/64, scales1/1/64,10–120K contexts, soft/
  coherent/edge data, changed cache/QKV/routes/lengths/positions/cache slots,
  and empty contexts. Vector has no separate fresh/sanitizer claim.
  Each comparison has14 shape/order entries,40 balanced AB/BA pairs each,
  independent reference/cache checks before and after capture. Three families
  give42 entries (including repeated parent controls).
  Sanitizers operate only native kernels plus torch independent FP32 partial
  reduction/output/cache checks; no Triton loader. Each tool/family executes
  24 calls (two orders, two pages, two scales, three changed-QKV calls).
  Final sanitizer audit and any source profiles are recorded in results/.

The generator deterministically reproduces all sources. No production source
or Triton plans change. Start the next session with the two prior pending H200
checks, not another sweep of these rejected L8 structures.

Final audit: scalar and prepare-once each have memcheck0 errors, synccheck0
errors, racecheck0 hazards/errors/warnings. All local processes completed.
SourceCounter profiles, B1/S33, run sequentially under the GPU lock after graph
comparisons/full benches and serialized against sanitizers: native LATE/EARLY
315056/311872 executed warp instructions; prepare-once attention192998 plus
preparation5104. Source-attributed excess shared wavefronts4104/4556/4564/0.
Long-scoreboard stalls remain distributed across query/normalization/gathers
in the fused kernels and waits in the prepared attention. No profiler duration
is used as a graph timing or an isolated causal latency estimate.
