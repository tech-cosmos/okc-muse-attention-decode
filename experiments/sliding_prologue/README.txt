Session 13: native single-tile sliding-decode prologue, 2026-10-24
================================================================

Production candidate: 5641ff1, source SHA256
1c7475a12eb2ca7dcb878b471c1b3c49e3b2d9bd5081e072a2914157d92a7d5b.
H200 check 4d36508dda044b58 was queued this session. Consult NOTES-codex.md and
profiles/session13-check-status.json for its latest state before scoring.

This experiment follows main session6's dependent-memory observation, with a
new native design rather than re-tuning the rejected session2 sliding kernel.
Every call computes all keys in the sliding window, updates new K/V as FP8,
and returns BF16 normalized/gated outputs. Queries may round only to E4M3.
No data-derived values are cached, and no work is moved outside kernel_fn.

One 128-thread CTA computes a 64-key tile of one KV head. Its cp.async copies
can start before norm/RoPE; the own token is computed independently, published
after the zero-filled gather completes, and attended in its natural tile.
This avoids a separate new-token QK/PV tile and any dependence on concurrent
cache writes. FP8 QK and FP16 P/V WGMMA accumulate in FP32. Only eight query
rows are useful; the WGMMA padding rows are zero. Each CTA emits FP32 partials.
The inherited Triton _combine_hb merges them under PDL. No Triton kernel or
plan was tuned. Native dispatch is sliding window2048, L1, B*Hkv<=4, G8, D64,
page1/64, RoPE present. Other shapes retain the production path.

Variants and generation
-----------------------
* generate_base.py / kernel.cu / native.py: Q built via shared memory;
  EARLY false/true controls gather timing, with the same arithmetic.
* direct_query.py / direct.cu / direct.py: build the useful query operands in
  registers using four lanes/head. Remove padded Q shared writes and Q reads.
  A different warp prepares the own K/V. EARLY true is the retained candidate.
* vector_query.py / vector.cu / vector.py: explicit uint2 query loads. Does not
  improve the target B1 case; not promoted. Has full-suite and scored comparison
  coverage, but no fresh/sanitizer claim.
* integrate.py embeds the exact direct.cu source into submission.py and adds the
  narrow shape dispatch. It expects the unchanged parent submission as input.
* production_parent.py is the exact pre-session submission, for comparisons.

Run in ~/okc venv from repository root. Generate with generate_base.py,
direct_query.py, then vector_query.py. The load_inline wrappers build locally
in ignored .build* directories. Production uses load_inline's normal cache.
The duplicate generate.py is the original standalone generator retained for
provenance; generate_base.py is used by the direct generator.

Local results
-------------
H100, cold-L2 CUDA graph replay, empty replay subtracted. Every custom timing
and profile driver takes /tmp/okc-gpu.lock over its complete GPU work. Exploratory
full benches use the inherited tools/bench.py locking (timing loops only).
No own profiler or validation process overlaps final comparisons.

Initial shared-Q native: B1 late6.112 / early5.888 us vs production4.832.
Direct-Q comparison: B1 late4.576 / early4.256 vs production4.672. Large batches
lose: direct early11.568/19.264 vs production10.400/15.760 at B16/B32.
Those values are from one same-process rotating/reversing comparison, not
cross-run performance ratios. Vector queries do not materially improve B1.

Final end-to-end integrated balanced adjacent AB/BA comparison (60 pairs each):
shape                         parent       candidate    paired delta
B1 ctx80000 page1              4.864 us     4.352 us       -0.432 us
B2 ctx80000 page1              5.376        5.024          -0.304
B1 ctx4000 page64              4.936        4.512          -0.472
B2 ctx4000 page64              5.416        5.136          -0.264
B1 ctx1024 page64              4.704        3.968          -0.760
Medians of per-variant means differ from medians of within-pair differences;
those columns need not subtract exactly. The B1 paired median speedup is1.099x.
Full integrated bench:1.916x vs fixed SGLang/H200 reference times, not a judge
score. Twelve nearby shape comparisons cover B1/2, contexts10–120K, pages1/64,
plus an unchanged B3 fallback control. Results/raw rounds are archived.

Validation and instruction evidence
-----------------------------------
* Shared-Q and direct-Q late/early: each16/16 full tests (3 native,13 fallback).
* Vector late/early: each16/16 full tests; comparison coverage only beyond that.
* Integrated submission:16/16 full tests,48 fresh replays (42 native,6 DP fallback).
* Four shared/direct variants:168 all-native fresh replays total, six changes
  each across seven nearby cases, including DP, both pages/scales, changed QKV,
  caches, routes, lengths, positions, slots, and empty contexts.
* Isolated native memcheck/synccheck:0 errors. Racecheck:0 hazards/errors/warnings.
  Each sanitizer checks48 calls across four variants, two pages, two scales,
  three fresh QKV calls. Torch independently reduces partials and checks output
  and cache against reference; no Triton loader involved.
* Base/direct/vector comparison drivers:12/20/20 variant-shape entries;
  nearby and integrated:24 each; balanced:10. Each checks output/cache before
  and after capture. These counts include repeated controls.
* audit.json verifies embedded production CUDA equals sanitized direct.cu.

All implementations use74 registers, zero stack/spills. Ptxas shared bytes
base20480 / direct16384; cuobjdump allocations include the additional1024 bytes
(base21504 / direct17408). Direct Q removes eight scalar LDS instructions and
six STS.U8 sites from the page1 static code. Matrix operation counts stay two
QGMMA and four HGMMA. Direct early's first LDGSTS is at0x6d0, before its first
RSQ at0x10d0; late's first LDGSTS is at0x1a00, after RSQ at0xb50.

Isolated NCU source counters at B1/S32: base early253768 executed instructions,
direct late163208, direct early164482. Shared excess wavefronts10248->4104.
Profiles support removal of query staging and the actual gather ordering.
Stall samples remain spread over memory, barriers, and constant/instruction
loads. Do not use profiler durations as graph timings or compare raw sampling
counts as precise latency estimates. All profiles ran after validation and
before the last balanced graph comparison; no overlap.

Scripts and logs
----------------
compare_base.py / compare.py / compare_vector.py: structural controls.
validate.py / validate_vector.py: full suite then full experimental bench.
fresh.py / integrated_fresh.py: changing input/routing graph replays.
sanitize.py: independent native partial/output oracle.
nearby.py / integrated_pairs.py / balanced.py: parent/candidate graph comparisons.
profile.py / summarize_profiles.py / inspect_sass.py: profiling and static audit.
results/: raw timings, logs, sanitizer reports, SASS/resources and summaries.

Do not re-run closed shared-query or vector-query sweeps. Any source change
requires a new exact-source qualifying check before a scored submission.
