# Session 35: transposed native full decode

Rejected full-attention L1 prototypes. Production remains byte-identical to the
frozen `production.py` (SHA256 62fefe79bd913e6dc1b12388ad7bec50139096cb07c7b61918b8031c0c93a86b).
Neither candidate is imported by production or sent to the judge.

The intended mechanism is fewer padded tensor-core rows and smaller score/output
fragments, while preserving 64-key paired-head gathers. This is a streaming
WGMMA design, unlike session28's large single-tile grid or main's warp-MMA tdec.
One 128-thread CTA handles a pair of KV heads. Three stages gather both full
128-byte K/V rows. Per head, QK is K[64,64] @ Q.T[64,8] with FP8 m64n8k32;
PV is V.T[64,64] @ P[64,8] with FP16 m64n8k16. Each thread holds four score
and four output floats per head. Max/sum reductions span all four warps through
shared memory. FP32 online softmax/partials, FP8 cache/query precision, every
key/value, cache insertion and the inherited PDL merge are preserved.

There is one fixed grid policy: ceil(528 / (batch * KV-head-pairs)), capped at
128 splits, with chunks balanced using each request's device-side length.
No split/stage/warp sweep or fitted dispatch cutoff was performed.

- `kernel.cu` / mode `transpose`: explicitly transpose V while converting to
  FP16, then use K-major A in PV.
- `natural_v.cu` / mode `natural`: packed conversion and vector stores preserve
  V's natural key-by-dimension layout; PV uses transA=1. `generate.py` derives
  exactly this conversion/descriptor/MMA change from the first source.

## Measurements

H100 complete cold-L2 graphs, 60 adjacent AB/BA pairs per case and candidate,
512 MB flush before every timed replay, empty replay subtraction. Each candidate
has its own paired production timings. Microseconds:

| Full decode | production / explicit transpose | delta | production / natural V | delta |
|---|---:|---:|---:|---:|
| B1, 80K, page1 |16.720 / 34.240|+17.600|17.056 / 22.112|+5.072|
| B8, 80K, page1 |73.840 / 149.264|+75.552|74.080 / 100.752|+26.496|
| B16, 80K, page1 |130.016 / 308.880|+178.816|129.856 / 185.296|+55.600|
| B32, 80K, page1 |241.888 / 570.544|+328.560|243.200 / 337.728|+94.672|
| B1, 80K, page64 |16.592 / 31.392|+14.720|16.480 / 19.072|+2.560|
| B4, 6K, page64, scale1/64, edges |11.120 / 12.240|+1.136|11.456 / 10.288|-1.184|
| B2, 9K, page64, mixed, soft |13.040 / 7.600|-5.472|13.440 / 5.984|-7.408|
| DP1, 4096, page64, scale1/64 |11.328 / 15.248|+3.824|11.584 / 12.096|+0.528|

Paired deltas need not equal differences of independent medians. The large
scored losses reject both variants. Incidental short-context gains do not
justify a cutoff or another short-context audit. Full-wrapper bench aggregates
1.440x / 1.750x versus fixed SGLang/H200 numbers are experimental local results,
not H200 estimates or production regressions.

## Mechanism evidence

Page1/page64 registers: explicit transpose 120/113; natural V 94/80. Both use
69,632 dynamic shared bytes (structure layout audited) plus 1,024 static bytes
reported by cuobjdump; no local/stack allocation. Each page has 4 QGMMA and
8 HGMMA static sites. See complete SASS/resources; these are not dynamic counts.

Two isolated NCU launches on B16/80K/page1, with independent output/cache checks:

| metric | explicit transpose | natural V |
|---|---:|---:|
| executed warp instructions |63,602,660|48,439,316|
| shared store bank conflicts |17,024,174|1,269,607|
| shared read bank conflicts |1,220|1,677|
| HBM read bytes |335,207,936|335,176,704|
| short-scoreboard stall % |48.67|22.46|
| long-scoreboard stall % |13.46|26.98|
| barrier stall % |3.00|5.80|

Eliminating the software transpose removes substantial work and bank conflicts;
it does not make the full kernel competitive. The counters do not isolate
latency or assign the remaining conflicts to a specific instruction. No profiler
duration, achieved-occupancy, or H200 inference is used as a speedup claim.

## Validation and reproduction

Activate `~/okc` and run from the repository root. `TDEC_VARIANT` is either
`transpose` (default) or `natural`, selecting an experimental module at import.

```
TDEC_VARIANT=natural python experiments/transposed_decode/smoke.py
TDEC_VARIANT=natural python experiments/transposed_decode/validate.py suite
TDEC_VARIANT=natural python experiments/transposed_decode/validate.py compare
TDEC_VARIANT=natural python experiments/transposed_decode/validate.py fresh
TDEC_VARIANT=natural /usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck --error-exitcode 1 python experiments/transposed_decode/sanitize.py
python experiments/transposed_decode/profile_parent.py
python experiments/transposed_decode/audit.py
```

- Both wrappers: 16/16 tests (5 actual-native, 11 fallback), followed by all
  12 benchmark shapes; 16 boundary smoke checks per variant, contexts0–4096,
  pages1/64.
- 960 adjacent graph pairs, 96 independent output/cache checks at timing
  checkpoints across eight shapes per variant.
- 64 actual-native fresh graph replays, plus 64 production controls. Four
  changed inputs per case include changed QKV/cache/routes/lengths/positions/
  cache slots, halved and empty contexts. TP8/DP, pages1/64, scales1/1/64,
  coherent/soft/edges, contexts4096–80K.
- Natural-V variant: memcheck0 errors, synccheck0 errors, racecheck0 hazards/
  errors/warnings; each tool checks24 fresh native graph replays. Torch reduces
  FP32 partials independently and validates output/cache. No Triton loader
  kernels are launched. No separate sanitizer claim for the explicit transpose.
- Custom comparison/fresh/smoke/sanitizer drivers hold the shared GPU lock over
  input creation/reference/capture/replay. Standard suite holds it for tests;
  its bench uses bench.py's timing-loop lock. All own GPU processes run in
  sequence. Profiling starts only after timing/fresh/sanitizer work finishes;
  the parent takes the lock before launching NCU and the child does not relock.
- CPU audit reproduces natural CUDA source, verifies production hash, output/
  gather/swizzle coverage, chunk coverage, raw pair medians, result counts,
  SASS sites and sanitizer/profile completion. Disassembly whitespace only
  was normalized after collection; no CUDA-token change.

`build.log` is an initial import log, not validation or timing evidence. All
final kernel runtime tests use dynamic shared storage. The wrapper later gained
frozen-production imports and an explicit variant selector; the CUDA sources
and native dispatch/arithmetic did not change after their runtime checks.

Close this cooperative transposed WGMMA topology as a scored optimization.
Another tile/stage/grid sweep is not justified by these large losses. A future
native design needs a concrete reduction in the remaining per-tile data movement
or synchronization, not the assumption that removing padded math is sufficient.
