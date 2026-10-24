Session19: native sliding L1 independent prologue loads (rejected).

Production parent: 3fb391f / SHA256 16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e.
base.cu exactly reproduces its embedded native attention; no Triton plans were tuned.

Baseline SourceCounters identifies page-result long-scoreboard wait before Q loads.
The two page indices already load together. qpre moves raw Q loads before gather;
plain uses ordinary C++ loads; rope also moves Q's eight cos/sin pairs. In SASS,
page-address calculation can still consume page results ahead of those Q loads.
ordered uses volatile Q and page loads to enforce issue order: Q loads do precede
page loads, but they become STRONG.SYS and latency gets worse. This control does
not isolate ordering from the changed memory semantics. All arithmetic/precision,
FP8 cache/query, FP16 PV, FP32 partials, new-token writes and PDL merge are unchanged.

H100 balanced adjacent AB/BA pairs, 60 rounds per shape, seed103:
                     B1 base/candidate us   paired delta   B2 delta
qpre                 4.240 / 4.256           0.000         -0.096
plain                4.320 / 4.336           0.000         -0.192
rope                 4.352 / 4.160          -0.192         -0.080
ordered              4.352 / 4.816          +0.512         +0.544

Rope confirmation, seed321 / 120 rounds, B1 4.480/4.320 but median paired delta
only -0.048 us. B2 ctx1024/page64 regresses +0.496 us. No reliable scored-shape
win sufficient to justify promotion/checking. B16/32 force the native control
which already loses to production; experimental broad dispatch is not proposed.
Every comparison checks independent output/cache before capture, after replay,
and after timing. The lock covers generation/reference/capture/replay; 512MB L2
flush before every graph; empty-replay cost subtracted. Do not infer small gains
from independent medians or independent runs.

All four: 16/16 local tests (3 native, 13 production fallback), then full bench;
60 all-native fresh replays per variant on 10 shapes, 240 total, including changed
values/routes/lengths/positions/cache slots and empty contexts; TP8/DP, pages1/64,
scales1/1/64, short/long contexts. rope and ordered: isolated memcheck/synccheck
zero errors, racecheck zero hazards/errors/warnings; 12 native calls per tool
and variant, independently torch-reduced FP32 partials/output/cache. No separate
sanitizer coverage for qpre/plain. Only EARLY=True has runtime coverage here.
Profiles baseline/qpre/rope/ordered are isolated under the same lock. Baseline
was profiled before variants; final profiles ran after correctness/timing work,
with no GPU overlap. NCU timing is not graph timing. No occupancy claim.

Reproduce from repo root (CUDA_HOME=/usr/local/cuda-13.0, ~/okc/bin/python):
  python experiments/sliding_loads/generate.py
  sh experiments/sliding_loads/run.sh
  sh experiments/sliding_loads/finish.sh
  LOAD_VARIANT=rope python experiments/sliding_loads/compare.py --seed 321 --rounds 120 --suffix=-confirm
  python experiments/sliding_loads/summarize.py
See scripts for isolated NCU/Compute Sanitizer invocations. Build modules before
profiling; use no concurrent GPU work. base/variant.cu and their bindings are
self-contained experiment assets, never imported by production.

production_before.py, integrated_pairs.py and integrated_fresh.py are for the
subsequent main merge + session16 dispatch integration, not these rejected variants.
Their integration results will be archived separately once run on merged production.
