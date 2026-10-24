# Native cooperative query preparation, session 33

Production snapshot: codex d289277, SHA256
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.

Session 32 measured long preparation intervals in warp zero of the native
single-tile sliding kernel. This experiment distributes its eight real query
heads across the four existing warps: sixteen lanes per head, four values per
lane, FP32 normalization/rotation, FP8 E4M3 rounding. All work stays inside the
timed call. No attention tiles, splits, dispatch boundaries, Triton/Gluon code,
cache precision, complete-key coverage, new-cache writes, or launch counts change.

Three implementations generated from the exact snapshot:

- `candidate.py`: compact 512-byte shared-memory handoff to the original warp-zero
  matrix fragments. Reuses an existing CTA barrier. Unlike session 13's shared
  query experiment, it does not stage any padded query rows.
- `swizzled.py`: same handoff with an XOR bank layout. Exhaustive CPU audit
  proves one producer/consumer per packed word and conflict-free stores/loads.
- `distributed.py`: two real query rows in each warp's WGMMA row block;
  warp-local shuffles populate the matrix fragments, eliminating the shared
  handoff entirely. Output stores restore the original contiguous head order.

All three versions lose to production across all sixteen tested shapes.
Initial `initial.json` uses `candidate` to mean the linear shared version; later
records also include an explicit `implementation` field. Each graph comparison
includes a separate identical-production capture to expose allocation/capture
offsets. Pair deltas are medians of adjacent AB/BA differences, not differences
of independent medians. Two seeds and reversed capture order are used for
confirmation. H100 paired candidate-minus-production deltas in microseconds,
initial / confirmation:

| sliding L1 | linear shared | swizzled shared | warp-local mapping |
|---|---:|---:|---:|
| B1,80K,page1 (scored) | +0.576 / +0.448 | +0.640 / +0.416 | +0.544 / +0.352 |
| B8,80K,page1 | +0.448 / +0.176 | +0.400 / +0.480 | +0.496 / +0.448 |
| B1,80K,page64 | +0.592 / +0.704 | +0.720 / +0.640 | +0.576 / +0.608 |
| DP1,4096,page64,scale1/64 | +0.240 / +0.320 | +0.240 / +0.192 | +0.192 / +0.096 |

No production change, dispatch adjustment or H200 request is justified. Small
nearby differences overlap identical-code capture offsets; this is not a precise
decomposition of preparation, shuffle or barrier cost. Static instruction counts
fall, but the real work now spans all four warps. Closing this mechanism avoids
another preparation/vector/layout parameter sweep. See final NOTES-codex.md
session 33 for the full validation and budget handoff.

Reproduce from the repo root in the existing venv:

```sh
python experiments/query_cooperative/generate.py
QUERY_VARIANT=distributed python experiments/query_cooperative/run.py suite
QUERY_VARIANT=distributed python experiments/query_cooperative/run.py compare --name distributed-initial --rounds 60
QUERY_VARIANT=distributed python experiments/query_cooperative/run.py compare --name distributed-confirmation --indices 0 1 3 4 7 8 12 13 --seed 33207 --reverse --rounds 120
QUERY_VARIANT=distributed python experiments/query_cooperative/run.py fresh --name distributed-fresh
QUERY_VARIANT=distributed compute-sanitizer --tool memcheck --error-exitcode 9 python experiments/query_cooperative/sanitize.py
python experiments/query_cooperative/audit.py
```

`run.py` holds the shared GPU lock around generation, references, capture and
all custom timing/fresh replays. Every timed graph has a 512MB cold-L2 flush;
an empty replay is subtracted. Standard suite/bench uses bench.py's timing-loop
lock. The first distributed suite completed all 16 tests, then was intentionally
terminated while waiting for that lock before its first timed benchmark; its
test log is preserved and the benchmark resumes separately. At that time the
swizzled profiler process was also waiting for the lock, not collecting counters.
That waiting profiler was terminated; `ncu-swizzled-deferred.log` contains no
measurements. Final profiling acquires the lock in the parent before starting
NCU, and child profiling avoids a second acquisition. The initial two successful
profiles use the earlier child-lock convention. Timing loops, custom fresh work,
sanitizers and actual profiler replays are serialized. Sanitizers use only the native binding plus
independent torch partial reduction, output normalization and reference checks;
they do not launch Triton loader kernels. NCU profiles follow timing/fresh work,
check complete output/cache, and are not used as graph timings. EARLY=True has
runtime coverage; EARLY=False is compiled only.

All generated sources, raw pairs, correctness logs, SASS, resources and profiler
metrics are archived. `audit.py` reproduces sources, checks unchanged Python
functions and production, enumerates packed-query/partial-store coverage,
recomputes pair medians, and verifies validation counts and static resources.
No judge jobs are requested by these scripts.
