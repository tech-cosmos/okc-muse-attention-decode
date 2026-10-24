# Session 32: native sliding phase-latency diagnosis

Offline diagnostic, not a proposed submission. Production SHA256 remains
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
No native/Triton/Gluon algorithm, plan or dispatch change was promoted. No judge
request. The experiment establishes a measured target before another kernel
rewrite, following the repeated rejected instruction/fusion experiments.

## What the probes measure

`generate.py` extracts the native sliding CUDA from the frozen production source.
Each compiled variant timestamps exactly one adjacent phase interval, using two
volatile `%clock64` reads in every thread and writing only each warp leader's
pair to a fresh, disjoint trace buffer at the end. No new barrier, cache access,
precision change, omitted key or output shortcut. The complete wrapper uses the
original native shape predicate and original PDL Triton merge. Unsupported cases
use frozen production. `run` returns `(output, trace)`; `kernel_fn` returns output.

Modes 0–9 cover metadata/gather issue, query/new-KV preparation, residual copy
wait, first CTA rendezvous, new-KV publication, V conversion/publication, QK,
softmax/FP16 probability packing, PV, and partial output. Mode 10 measures the
whole instrumented body. Mode 11 has two adjacent clock reads. Mode -1 compiles
out clocks and trace writes. Both pages' mode -1 SASS is instruction-for-instruction
identical to production (1056/1064 decoded instruction sites). A second capture
of the *actual production function* supplies a separate identical-code control.

SASS audit verifies clock placements, including: exactly one `DEPBAR.LE SB0,0`
between the copy-wait clocks; both QGMMA and their wait in QK; four HGMMA and their
wait in PV. Clock-pair control has no instruction between reads and reports
2 cycles. Registers stay 74 except QK-interval mode at 75, all have 17408 shared
bytes and zero stack/local allocation. See `clock-placement.json`, full SASS,
and compiler resource reports. Lineinfo and extra unused parameters do not
change the no-clock control's decoded instructions.

## Results and limits

H100, cold-L2 complete CUDA graphs. Initial seven shapes, 40 adjacent AB/BA pairs
per contrast; confirmation five shapes, second seed/reversed graph capture order,
80 pairs per contrast. Every graph computes and checks real output/cache values.
Trace snapshots are a **separate** series of 12/20 cold replays per interval.

Representative median **warp-zero cycles**, initial / confirmation:

| Interval | B1 80K page1 | B8 80K page1 | B1 80K page64 |
|---|---:|---:|---:|
| metadata + gather issue | 1726 / 1611 | 1833 / 1875 | 1596 / 1579 |
| query + new-KV preparation | 2448 / 2401 | 5662 / 5573 | 2474 / 2554 |
| residual asynchronous copy wait | 5 / 5 | 5 / 5 | 5 / 5 |
| first CTA rendezvous | 21 / 21 | 22 / 23 | 21 / 21 |
| value conversion/publication | 347 / 347 | 408 / 403 | 339 / 339 |
| QK matrix + wait | 128 / 128 | 177 / 173 | 128 / 128 |
| softmax/probability packing | 657 / 656 | 769 / 770 | 658 / 658 |
| PV matrix + wait | 182 / 182 | 251 / 247 | 182 / 182 |
| whole body | 5674 / 5697 | 9960 / 10100 | 5512 / 5631 |

Warp zero carries the eight real query rows. Other warps reach the copy wait
sooner and wait longer: initial B1 page1 about1530–1576 cycles versus warp-zero5.
This is evidence of preparation/copy overlap, not evidence that gathering is free.
The short-context new-token owner behaves differently: at B1ctx10, its new-KV
warp spends~850 cycles in publication and warp0 spends~1140 at V publication.
Raw samples and `summary.json` retain owner/non-owner and per-warp separation.

Unmodified-production NCU corroboration (one isolated launch per shape):
B1page1/B8page1/B1page64 long-scoreboard samples55/207/52 out of176/595/167,
barrier22/88/21; executed warp instructions164482/1315856/167042.
Achieved active-warp percentages5.96/21.52/5.96. These small sampling counts support
only a broad dependency/scheduling interpretation. The CSV retains every PC and
counter; samples are not precise percentages of graph latency. Profiler duration
is never substituted for graph timing. No isolated cache-latency attribution.

**Conclusion:** native small sliding spends substantial time before its matrix
work. On the real-query warp, outstanding copies are already covered by preparation
in the measured long-context cases. Shaving a few conversion/matrix instructions
or merely moving their waits is not a demonstrated path to a useful gain. A future
attempt needs to reduce the metadata/preparation dependency path or redundant work
without paying a larger launch/synchronization cost. Session19's failed query-load
ordering controls and session14's failed L8 preparation kernel remain closed;
these traces do not justify repeating them or fitting a cutoff.

Do **not** sum interval medians or treat them as an additive model of original
production: each interval comes from a different compiled probe and separate
replay. Their scheduling and late trace stores can perturb execution. Initial
whole-body probe delta is+0.272us at B1page1, confirmation+0.032; same-production
capture controls range-0.256 to+0.160us across the two runs. No-clock identical-SASS
control offsets reach+0.224/-0.176us. Thus graph allocation/capture effects and
instrumentation are material for small differences. No speedup claim, no
percentage of latency claimed removable, and no H200 extrapolation. Suite bench
1.982x versus fixed SGLang/H200 numbers is diagnostic-wrapper evidence only.

## Validation and reproduction

- Diagnostic mode10 wrapper: 16/16 full tests (3 actual-native,13 fallback), then
  all12 benchmark shapes.
- 9520 adjacent graph pairs; 540 independent reference output/cache checks at
  comparison checkpoints. Seven shapes cover B1/3/8,TP8/DP,pages1/64,scale1/1/64,
  soft/coherent/edges,contexts10/2047/4096/80K. Five second-seed confirmations.
- 2208 separate cold trace replays, four warp leaders per CTA.
- 364 actual-native fresh graph replays: seven shapes x13 modes x4 new inputs.
  QKV/cache bytes/page routes/sequence lengths/positions/cache slots change;
  final two inputs shorten or empty contexts. All output/cache references pass.
- Three unmodified production NCU launches with independent output/cache checks.
- No sanitizer run, and no new sanitizer coverage claimed. Synchronization is
  unchanged; CPU audit enumerates the trace indexing for all split extents1–32
  and used batch/head shapes, proving writes fit and are disjoint.
- Custom drivers lock input generation/reference/capture/timing and profiling
  with `/tmp/okc-gpu.lock`. Every timed/trace replay flushes512MB; graph timing
  subtracts an empty replay. Suite/bench keeps bench.py's timing-loop lock and
  ran without overlapping own GPU work. All local processes finished.

Run from repo root with `/home/daytona/okc/bin/activate`:

```
python experiments/phase_latency/generate.py
python experiments/phase_latency/run.py suite
python experiments/phase_latency/run.py measure
python experiments/phase_latency/run.py measure --reverse --seed 32091 --rounds 80 --samples 20 --indices 0 1 2 4 5 --name confirmation
python experiments/phase_latency/run.py fresh --name fresh
python experiments/phase_latency/collect_profiles.py
python experiments/phase_latency/audit.py
```

Redirect logs to their matching paths in `results` as in the archive. NCU binaries
are excluded; exported complete source/metric CSVs are retained. Audit reproduces
sources, checks frozen production identity, counts, paired medians, trace bounds,
SASS clock placement/identity and profile completion. SASS/resource text and the generated CUDA archive only had
trailing whitespace/EOF normalization after testing (no token changes). No diagnostic import enters production.
