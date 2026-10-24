Session 18: native single-tile sliding QK/V scheduling
===================================================

Rejected local experiment. Production submission.py is unchanged (source3fb391f,
SHA256 16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e).
This tests the latency-bound L1 single-tile kernel, not session12's full-attention
producer/consumer loop. It changes no Triton plans or kernel code.

The production order is V conversion -> QK -> softmax -> PV. generate.py extracts
its exact native CUDA from submission.py and produces:
  overlap: issue/commit QK, convert/fence/publish V, then wait for QK.
  late: wait for QK, softmax, then convert/fence/publish V before PV.
  pinned: overlap plus explicit post-wait compiler liveness for register Q inputs.
The pinned and overlap page1/EARLY instruction lines are identical. Keep pinned
as the conservative source if reusing this experiment. The other pages/EARLY
modes are compiled; only EARLY=True is runtime-validated in this cycle.

New-token K writes retain an async-proxy fence before the existing CTA barrier.
V writes retain their own fence/barrier. All matrix operations remain converged,
QK scores stay compiler-pinned until completion, and no K/V operand is reused
before its matrix operation completes. Every cached/new key is attended, with
unchanged FP8 cache/query precision, FP16 P/V and FP32 partials. The existing
production PDL merge remains. No output cache or input-derived setup is added.

candidate.py forces native broadly for L1 sliding G8/D64/page1-or64 to measure
nearby shapes; this is NOT proposed production dispatch. In particular, forced
native B16/32 already loses to production. Apparent tiny improvements over those
native controls do not establish a production win.

Run from repo root with ~/okc activated and CUDA_HOME=/usr/local/cuda-13.0:
  python experiments/sliding_qk_overlap/generate.py
  OVERLAP_VARIANT=pinned python experiments/sliding_qk_overlap/validate.py
  OVERLAP_VARIANT=pinned python experiments/sliding_qk_overlap/bench.py
  OVERLAP_VARIANT=pinned python experiments/sliding_qk_overlap/compare.py
  OVERLAP_VARIANT=pinned python experiments/sliding_qk_overlap/fresh.py
Repeat as appropriate with overlap/late. Both pinned and late have fresh-input
and native-only sanitizer coverage; overlap has full-suite and graph comparison
coverage. Sanitizers run independently reduced output/cache reference checks:
  OVERLAP_VARIANT=pinned /usr/local/cuda-13.0/bin/compute-sanitizer \
    --tool memcheck --error-exitcode 99 python experiments/sliding_qk_overlap/sanitize.py
Repeat memcheck/synccheck/racecheck for pinned and late. These avoid the unrelated
Triton loader attribute probe. Each tool/variant runs12 changed-QKV native calls,
page1/64, scales1/1/64, B6 ctx2048. Fresh driver runs60 replays/variant across10
shapes, changing QKV/cache/routes/lengths/positions/cache slots, including empty
contexts; TP8/DP, pages1/64, scales1/1/64, contexts10-120K, varied distributions.

Every custom GPU driver holds /tmp/okc-gpu.lock during generation, reference
checks, capture and replay. Comparisons use60 adjacent balanced AB/BA pairs per
shape,512MB L2 flush before every replay, and empty-graph subtraction. Each output
and cache is independently checked before capture, after replay, and after timing.
Full benches retain tools/bench.py's timing-loop lock and run sequentially with
our other GPU processes. No profiler is used.

Results: H100 microseconds, independently paired runs
----------------------------------------------------
variant       B1 parent/candidate (paired delta)    B2 parent/candidate (delta)
overlap       4.464 / 4.480 ( 0.000)                5.024 / 4.992 (-0.048)
late          4.320 / 4.320 ( 0.000)                5.088 / 5.056 (-0.016)
pinned        4.384 / 4.384 (-0.016)                5.024 / 4.960 (-0.096)

None gives a convincing scored-shape win. Nearby B3/8 and short B1 can regress.
Do not infer a new dispatch from small differences. Each variant passes16/16
(3 native,13 fallback) and its full bench. Bench aggregates1.909/1.899/1.895x are
rejected broad-native experiments, not production regressions or judge results.

SASS confirms real rescheduling: overlap/pinned's QGMMA precedes four V STS.128
and QK wait follows them. Late's QK wait precedes V stores, which ptxas interleaves
with softmax. All have74 registers,17408 shared bytes (cuobjdump; ptxas16384),
zero stack/local allocation and spills. Static page1/EARLY instructions1049 base
vs1045 candidates; same2 QGMMA,4 HGMMA,32 EX2 and3 CTA barrier sites. These are
static instruction sites, not dynamic counts or achieved-occupancy measurements.

After building all variants:
  python experiments/sliding_qk_overlap/inspect_sass.py
  python experiments/sliding_qk_overlap/audit.py
The audit reproduces sources and checks completed test, bench, raw-pair, fresh,
sanitizer and SASS reports. No new check or scored run was requested. Existing
scored job71e2eaa7b52c407f belongs to unchanged production and must not be resubmitted.
