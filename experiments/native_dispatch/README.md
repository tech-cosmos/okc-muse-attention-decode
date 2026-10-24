# Session 25: native dispatch generalization audit

2026-10-24. Production is unchanged. This is an offline comparison of complete
kernels, not a new attention implementation or a judge submission. Frozen main
snapshot: `3daf4a4`; production SHA256:
`6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf`.

The preceding native micro-optimization families are closed. This cycle checks
whether the existing native dispatches still improve on current main across
nearby shapes, including their boundaries. Native sliding is selected through
B*Hkv=16; small-full native reduction additionally requires 64<S<=128. The
Triton/Gluon implementations, plans and launches were not changed. The common
14 top-level helper function ASTs match the frozen main snapshot exactly.

## Measurements

64 distinct shapes, 80 adjacent AB/BA pairs per shape, followed by eight
preselected suspect/neighbor/control shapes at seed25104 with 120 pairs each.
Initial seed25103. All measurements use CUDA graph replay on the H100, a 512 MB
L2 flush immediately before each timed replay, and empty-replay subtraction.
The GPU lock covers generation, reference checking, capture and timing.
Raw triplets are main/codex/empty microseconds. A negative delta favors Codex.

| Selected shape | main / codex us | paired delta us |
|---|---:|---:|
| Sliding L1 B1, 80K page1 fixed | 4.832 / 4.416 | -0.416 |
| Sliding L1 B8, 80K page1 agentic | 7.808 / 7.536 | -0.224 |
| Sliding L1 B8, 80K page64 agentic | 7.872 / 7.264 | -0.624 |
| Full L1 B1, 80K page1 fixed | 17.808 / 17.056 | -0.720 |
| Full L8 B1, 80K page1 fixed | 19.168 / 18.272 | -0.848 |
| Full L8 B2, 80K page1 agentic | 38.448 / 37.712 | -0.752 |
| Full L8 B2, 9K page64 agentic | 10.016 / 10.192 | +0.192 |
| Same 9K shape, second seed | 10.048 / 10.304 | +0.256 |

All 24 native-sliding cases favor Codex, with paired changes -1.376 to -0.192 us.
They include TP8/DP, pages1/64, scales1/1/64, contexts63-120K, mixed lengths,
B7/B8 near the boundary, and short contexts around tile/window boundaries.
The four sliding fallback comparisons just beyond the boundary are controls.
The complete scan has 16 unchanged-fallback cases, with deltas -0.064 to +0.256 us;
those differences are measurement variation, not kernel differences or wins.

The 24 native-full cases span changes -1.920 to +0.192 us in the first scan.
The B2/L8/9K/page64 exception repeats at +0.256 us, about 2.5%, with S76.
The within-run block-bootstrap intervals are [0.112,0.304] and [0.216,0.304] us.
Nearby B2/L8/S65/page64 is -0.160/+0.032 initially and 0.000/-0.144 on confirmation;
page1 S65 consistently wins. B2/L8/40K/page64 remains faster by1.584 us on
confirmation. These points do not identify a justified broad dispatch cutoff.
No unconditional "Codex is never slower than main" claim is supported. Record
this narrow nearby weakness for a targeted follow-up; do not retune native merge
architecture or main's plans from this small result.

Exact B1 full S64/S65 transitions are checked at ctx8191/8192 for L1 (both pages)
and ctx8184/8185 for L8 (page1). The new token block counts toward capacity.
L8 probes at ctx8191/8192 both have S65 and are labeled as nearby probes, not
as the exact transition. Native-side paired deltas at the exact transition are
-0.480/-0.256 us for L1 page1/page64, -0.720 us for L8 page1. There is no measured
native regression at those exact transitions. Do not infer changes in a single
implementation by subtracting times across different shape runs.

Paired deltas need not equal differences between separate medians. Additional
intervals bootstrap adjacent AB/BA blocks (mean delta within each block, median
across blocks); they describe within-run variation only. They do not account for
all system variation or multiple comparisons, and are not H200/judge intervals.
No aggregate improvement, new production speedup, or profiler-duration claim.

## Validation and an offline driver correction

All 72 timed entries (64 shapes plus eight confirmations) independently check
both complete implementations before capture, after first replay and after
timing: 432 output/cache reference checks. No incomplete-output path is used.

24 shapes also replay both implementations on four fresh inputs, including
changed QKV, caches, page tables, lengths, positions and new cache slots. The
last two shorten generated requests and include empty contexts. All192 corrected
fresh graph replays pass. Codex's96 comprise28 native-sliding,40 native-full and
28 fallback calls; main's96 are separate controls. This is independent reference
validation, not a new sanitizer claim. Native source is unchanged and retains
its earlier sanitizer coverage; no redundant unchanged 16-case suite was run.

The initial fresh driver aborted with an illegal memory access after the first
shape. Its original length mutation used randint(0,ctx+1) regardless of each
newly generated request's allocated length. On agentic inputs that can extend a
request into page-table entries that were never assigned. The corrected driver
only shortens each generated request, checks nonnegative new cache rows, and
logs the exact replay context. Both sources then pass all192 replays. The
original failing random draw was not recorded, so this is an identified driver
validity defect and successful corrected validation, not a definitive attribution
of that particular GPU fault. The failed log and partial result are preserved as
`fresh-invalid-lengths.*` and are excluded from pass counts. The timed comparisons
never mutate lengths and are unaffected by this driver correction.

## Reproduce

From the repository root in the existing `~/okc` environment:

```sh
python experiments/native_dispatch/audit_dispatch.py compare --indices $(seq 0 49)
python experiments/native_dispatch/audit_dispatch.py compare --indices $(seq 50 61) --name split-boundary
python experiments/native_dispatch/audit_dispatch.py compare --indices 62 63 --name verify-split-boundary
python experiments/native_dispatch/audit_dispatch.py compare --indices 36 41 43 49 58 59 60 61 --seed 25104 --rounds 120 --name confirmation
python experiments/native_dispatch/audit_dispatch.py fresh
python experiments/native_dispatch/summarize.py
```

Each command must finish before the next. Redirect stdout/stderr to the matching
`results/*.log` if rebuilding that evidence. The driver defaults to all64 shapes
now; the explicit first50 indices reproduce the original scan file. The CPU-only
audit checks source identity, unchanged helper ASTs, complete result/pair counts
and fresh replay counts. No experimental file is imported by production.
