# Deferred FP32 denominator reductions — session 36

This experiment improves session35's natural-V native transposed full-decode
prototype, but still loses to production on all three scored full-decode cases.
It is not imported or dispatched by submission.py. No judge requests.

## Mechanism

For one query, every lane uses the identical running maximum m and rescale
alpha. If local contributions are l_i, the recurrence
`l_i <- alpha * l_i + sum(local probabilities)` can be reduced across lanes
once at exit: summation commutes with multiplication by the shared alpha.
All arithmetic stays FP32; this changes association, not dtype or precision.

- `natural_v.cu`: exact session35 natural-V source (fded6bf).
- `deferred.cu`: retain a warp-local denominator; move the cross-warp sum
  out of the tile loop. Remove one per-tile stats-reuse barrier and shared
  sum traffic; add an epilogue publication/barrier/reduction.
- `lane.cu`: also defer the within-warp sum shuffles to that epilogue.

Same FP8 Q/K, FP16 P/V, FP32 accumulators/partials, 64-key tiles, three gather
stages, split policy, new-cache writes and PDL merge. Every key/value remains.
The numerator and maximum code are unchanged. No tile/stage/grid sweep.
`generate.py` reproduces both candidates from the frozen natural source.

## Complete graph results

H100 microseconds, 60 adjacent AB/BA pairs per case/contrast; each contrast has
its own baseline capture. Negative paired deltas indicate improvement.

| B / 80K / page1 full L1 | production / lane | lane minus production | natural / lane | lane minus natural |
|---|---:|---:|---:|---:|
| 1 | 16.992 / 20.912 | +3.984 | 21.904 / 20.880 | -1.104 |
| 16 | 128.736 / 173.072 | +44.192 | 181.808 / 173.040 | -8.224 |
| 32 | 239.904 / 315.744 | +76.000 | 333.440 / 315.312 | -18.336 |

Cross-warp-only deferral saves 0.768/4.352/8.400 us against natural at B1/16/32,
but loses 4.176/48.528/86.736 us against production. Nearby short-context gains
remain incidental; no fitted cutoff or H200 check is justified. Raw results
cover eight shapes per variant/control, including pages1/64, DP, scales1/1/64,
soft/coherent/edges. Paired deltas need not equal separate-median differences.
Broad-wrapper bench aggregates: deferred1.760, lane1.788 versus fixed
SGLang/H200 times, not production regressions or H200 estimates.

## Evidence

- Both wrappers16/16, five actual-native and eleven fallback cases; each then
  completes all12 scored benchmark shapes.
- 1920 adjacent graph pairs; 192 independent output/cache reference checks.
- 64 fresh native graph replays plus64 separate production controls, four
  changed inputs per case and variant. QKV/cache/routes/lengths/positions/cache
  slots change, including shortened and empty contexts.
- 32 direct native partial comparisons, through context1,048,576, both pages
  and scales. Numerators/maxima/cache are bit-exact against the natural control;
  largest denominator relative difference2.088428e-6. Long recurrence checks
  compare partials; they are not independent million-token output references.
- Each candidate independently: memcheck0errors, synccheck0errors,
  racecheck0hazards/errors/warnings. Each tool runs24fresh native graph replays
  and independently checks Torch-reduced outputs and caches. No Triton loader.
- Lane has16additional boundary smoke checks, contexts0/1/10/63/64/65/257/4096,
  pages1/64. Cross-warp-only has no separate boundary-smoke claim.
- Custom drivers hold the shared GPU lock through generation/reference/capture/
  replay; every timed replay flushes512MB and subtracts empty graph cost.
  Standard bench keeps bench.py's timing-loop lock. All own GPU work runs
  sequentially. NCU parent takes the lock before launching each profiler child.

Page1/page64 registers stay94/80 for all three kernels, with69,632dynamic shared
bytes plus1,024static, zero stack/local. Page1 static instruction sites
natural/deferred/lane1362/1354/1362, all4QGMMA/8HGMMA and54SHFL sites. Static
site counts conceal moving instructions out of loops.

Three separately reference-checked B16/page1 NCU launches execute
47,833,256 /45,110,464 /43,344,312 warp instructions. HBM reads
330,934,016 /330,932,480 /330,930,944 bytes. Shared-store bank conflict counts
1,219,801 /1,175,255 /1,241,935 do not explain the speedup; the source and dynamic
instruction count support removing repeated reduction work. Barrier stall
percentages5.77/5.97/6.58 do not measure isolated barrier latency. No profiler
latency or achieved-occupancy speedup claim.

`audit.py` checks source identity/reproduction, results/counts, raw paired
medians, resources and profiler logs. Frozen production is9dbf51a/SHA62fefe;
the later main BF16 integration belongs to a separate production validation.

## Reproduce

Activate `/home/daytona/okc/bin/activate`, then from repo root:

```sh
python experiments/deferred_denominator/generate.py
TDEC_VARIANT=lane python experiments/deferred_denominator/validate.py suite
TDEC_VARIANT=lane python experiments/deferred_denominator/validate.py compare
CONTROL=natural TDEC_VARIANT=lane python experiments/deferred_denominator/validate.py compare
TDEC_VARIANT=lane python experiments/deferred_denominator/validate.py fresh
python experiments/deferred_denominator/partials.py
TDEC_VARIANT=lane compute-sanitizer --tool racecheck --error-exitcode 99 python experiments/deferred_denominator/sanitize.py
python experiments/deferred_denominator/profile_parent.py
python experiments/deferred_denominator/audit.py
```

Repeat candidate commands with`TDEC_VARIANT=deferred`; sanitizer tools are
memcheck/synccheck/racecheck. Build at import, no input-derived cache.
Close this denominator-deferral cycle. Its real instruction reduction does not
close the scored gap; no further reduction or tile/grid parameter sweep follows.
