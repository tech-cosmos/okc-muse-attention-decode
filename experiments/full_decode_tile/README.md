# Session 28: native full-decode single-tile attention

Rejected on complete CUDA graphs; no production change or judge request.
Frozen production d8bab0a SHA256 `1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
The main snapshot is 824e52a. No Triton/Gluon plans or source were tuned.

## Mechanism

A new full-attention L1 kernel uses one 128-thread CTA per 128 or 256 keys,
with all gathers issued before query normalization. There is no streaming
attention loop, running softmax, or intra-loop pipeline. FP8 QK uses
m64n128/m64n256 WGMMA; PV uses FP16 m64n64k16 WGMMA over every key.
Only eight query rows are useful. The kernel normalizes and writes the new
FP8 K/V token in its natural tile. FP32 partials feed the unchanged inherited
Triton merge, with PDL. All input-derived work is inside each timed call.

`generate.py` derives CUDA from the frozen native sliding source, removes
RoPE/sliding bounds, expands the key dimension, explicitly keeps asynchronous
register probabilities alive through the final wait, and uses disjoint dynamic
shared buffers. `candidate.py` is an offline broad L1/full wrapper; it is not
imported into production. The broad wrapper exposes losses at large batches.

## Complete graph evidence

H100, 60 adjacent AB/BA pairs for each shape/width. Each width has its own
paired parent control. Flush 512 MB before every graph, subtract empty replay;
GPU lock covers generation, reference checks, capture and timing. Raw pairs
are in `results/initial.json`; paired deltas are not differences of medians.

| full L1 | parent / tile128 us | paired delta | parent / tile256 us | paired delta |
|---|---:|---:|---:|---:|
| B1 80K page1 |16.976 / 22.944|+6.000|17.040 / 25.584|+8.560|
| B2 80K agentic page1 |28.336 / 41.968|+13.440|28.432 / 44.064|+15.600|
| B16 80K agentic page1 |129.920 / 229.552|+99.600|129.760 / 256.624|+126.752|
| B32 80K agentic page1 |240.928 / 448.016|+206.816|240.848 / 501.904|+261.248|
| B1 80K page64 |16.720 / 22.672|+6.032|16.640 / 23.936|+7.200|
| B2 9K page64, scale1/64 |11.088 / 8.544|-2.608|11.184 / 10.048|-1.136|
| B1 10 page1 |5.072 / 4.448|-0.608|5.072 / 5.824|+0.704|
| DP1 4096 page64, scale1/64 |11.584 / 12.528|+1.056|11.504 / 14.848|+3.360|

Large scored losses are decisive; no repeated favorable-seed selection or
parameter sweep. Incidental short-context wins are not a validated dispatch
rule. Full broad-wrapper benches are 1.619x/1.541x versus fixed SGLang/H200
numbers, not production regressions or H200 speedup estimates.

Page1/EARLY compiler registers: 112/224; dynamic shared memory: 32768/65536
bytes, plus cuobjdump's 1024 static bytes. Both have zero stack/local memory
and spills. SASS counts: two QGMMA each; 8/16 HGMMA, LDGSTS, vector V stores
and raw-V loads. No NCU or achieved-occupancy claim. Larger independent tiles
trade away pipeline work for wider matrix/softmax work, more partials and
register pressure; these measurements do not isolate each cost.

## Correctness and a generator bug

Initial 128-key source was WRONG. Broad sequential string replacements first
changed the gather loop from 2 to 4, then unintentionally changed it from 4 to
8 when widening the conversion loop. It copied 256 rows into 128-row buffers,
overwriting adjacent shared arrays. Output checks caught it before timing.
QK scores/maxima matched; V/PV were corrupt. Probability-register pinning and
synchronous PV controls did not fix the bug. The corrected generator scopes
the conversion replacement and the CPU audit checks exact gather byte capacity.
Final partial/output checks at contexts 10/63/64/127/128/255/256/1024 pass.

`invalid-*` sources, the pre-fix debug logs and `debug-memcheck.log` are failure
evidence ONLY. That early memcheck reported zero address errors while numerical
checks failed: it did not detect the overlap of logical arrays inside shared
allocation. It is not correctness/sanitizer coverage of final code. The old
intermediate dump logs also belong to incorrect source. No failed source was
benchmarked or submitted. No withdrawal is needed.

Final corrected code:

- Both 16-case suites pass (5 actual-native, 11 fallback tests per width), each
  followed by the full 12-shape benchmark.
- 72 independent output/cache checks in 8-shape/3-mode comparisons, before
  capture, after first replay and after timing; 960 adjacent graph pairs.
- 48 actual-native fresh graph replays: 6 shapes, 4 changed inputs, 2 widths.
  QKV, caches, routes, lengths, positions and cache locations change; shortened
  and empty contexts, DP, pages1/64 and scales1/1/64 are covered.
- Native-only memcheck 0 errors, synccheck 0 errors, racecheck 0 hazards/errors/
  warnings. Each checks 24 fresh graph replays against an independent Torch
  FP32-partial reduction and output/cache reference. No Triton loader kernels.
- Eight final boundary output/cache checks and independent partial comparisons.
- All GPU work serialized through the shared lock, except standard bench.py
  warmup/reference phases retain that driver's original lock semantics. None
  overlaps another GPU process launched by this session.

## Reproduction

From repo root, activate `/home/daytona/okc/bin/activate`:

```sh
python experiments/full_decode_tile/generate.py
python experiments/full_decode_tile/run.py compare --rounds 60
python experiments/full_decode_tile/run.py suite
python experiments/full_decode_tile/run.py fresh --name fresh
python experiments/full_decode_tile/debug.py
/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 99 python experiments/full_decode_tile/sanitize.py
/usr/local/cuda-13.0/bin/compute-sanitizer --tool synccheck --error-exitcode 99 python experiments/full_decode_tile/sanitize.py
/usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck --error-exitcode 99 python experiments/full_decode_tile/sanitize.py
python experiments/full_decode_tile/audit.py
```

Redirect to the correspondingly named `results/*.log` files for the audit.
Do not run the invalid generator; its purpose is to preserve the failure.
Production retains its previous validation; no unchanged-production rerun or
new production sanitizer coverage is claimed.
