Session 16: native sliding decode boundary and padded softmax rows
================================================================
Production submission.py is UNCHANGED from 3fb391f (SHA256 16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e).
No Triton source or plans were tuned. None of the experimental files are imported
by production. No new check or scored request was made for these experiments.

1. Boundary survey (a useful nearby-shape result, not a scored-load improvement)
----------------------------------------------------------------------------
candidate.native reuses the current native CUDA attention and inherited merge,
with the identical arithmetic, scratch layout and launch configuration. It forces
that path on supported L1 sliding shapes for comparison. candidate.kernel_fn is
the proposed conservative shape-only dispatch: B*Hkv <=16, otherwise production.
proposed-dispatch.patch is the corresponding production edit, deliberately NOT
applied while the exact-current-source H200 check is pending. It should be bundled
with a future qualified source change rather than spend a judge check solely on
an unscored boundary change. No H200 claim is made for the broader dispatch.

Initial and second-seed confirmation at 80K/page1, microseconds:
 B       initial parent/native      confirmation parent/native    paired delta
 3            6.336/5.408                 6.336/5.312                 -1.088
 4            6.528/5.744                 6.432/5.536                 -0.912
 5            6.848/6.112                 6.976/6.144                 -0.768
 6            7.360/6.688                 7.296/6.624                 -0.688
 8            7.744/7.456                 7.808/7.424                 -0.384
Initial B12/16/32 native loses (9.760/11.552/19.296 vs9.200/10.304/15.664).
Nearby scans include short/mixed contexts, pages1/64, edge/coherent inputs,
1/64 scales and DP heads. DP1 (=16 KV heads) wins; DP2 at long context loses.
The final edge scan includes B7/8/9/10 on both pages; inspect edge-base.json.
The threshold16 is conservative, not a claim that every shape above it loses.

2. Padded-row structural controls (rejected)
-------------------------------------------
generate.py extracts the exact production CUDA into base.cu and replaces only
softmax/probability packing. QK/PV WGMMA, every historical key and own token,
cache writes, precision, split count and output merge are preserved.
pruned.cu computes probabilities for warp zero's eight real output rows only;
all warpgroup threads reconverge for collective PV. Other query rows are padding,
not omitted actual queries/keys. half.cu removes the second eight padded rows
in each warp without the warp-zero branch. Other warps compute unused rows.

Balanced structural comparisons against the ORIGINAL NATIVE path:
 shape                 base/pruned       base/half
 B1 80K page1          4.464/4.432        4.272/4.272
 B2 80K page1          4.976/4.800        4.928/4.848
 B16 80K page1        11.536/11.920      11.520/11.200
 B32 80K page1        19.312/20.720      19.312/19.232
These are separate runs; compare each candidate to its own control. B1 paired
deltas -0.064/0.000us do not justify promotion; neither closes the large-batch
gap to Triton. Raw pairs, all nearby entries, and initial comparisons are saved.

Static early/page1 SASS: base/pruned/half registers74/106/74; shared bytes17408
(cuobjdump),16384(ptxas), zero stack/local memory and spills. Instruction sites
1049/1185/945; EX2 sites32/32/16; F2FP52/44/44; matrix ops remain2 QGMMA +4 HGMMA.
Pruned has more branches and compiler-duplicated instruction sites; counts are
STATIC, not executed-instruction counts. No NCU profiling or occupancy claim.
Reducing source arithmetic and static instruction sites did not establish a B1 win.

Validation and timing
---------------------
- Full16-case reference suites for both structural variants (3 native/13 fallback),
  each followed by broad-native full bench:1.886x/1.892x against fixed SGLang-H200
  numbers. These are rejected experimental aggregates, not production results.
- Proposed boundary wrapper independently passes16/16, with its own full bench.
- base/pruned/half each60 all-native fresh graph replays (10 shapes x6 seeds):
  TP8/DP, pages1/64, scales1 and1/64, soft/coherent/edges, ctx10–120K; changed
  QKV/cache/routes/lengths/positions/cache slots and empty contexts.
- pruned and half each isolated memcheck/synccheck/racecheck:12 native calls per
  tool (B6,ctx2048,two pages,two scales,three changing-QKV calls). Independent
  torch partial reduction/output and cache checks; no Triton loader. See logs
  and audit.json for completion; only zero-error completed reports count.
- Every custom comparison holds /tmp/okc-gpu.lock across generation, references,
  capture and timing;512MB flush per replay, empty graph subtraction,60 adjacent
  balanced AB/BA pairs per entry. References before capture, after initial replay,
  and after timing. Full bench uses bench.py's lock; it runs after our other GPU
  work completes. No profiler overlaps timings. No eager latency claims.
- EARLY=True only is runtime validated here; compiled EARLY=False is unused.

Reproduce from repository root with /home/daytona/okc/bin/python:
 python experiments/sliding_boundary/generate.py
 python experiments/sliding_boundary/validate.py
 python experiments/sliding_boundary/bench.py
 python experiments/sliding_boundary/validate_half.py
 python experiments/sliding_boundary/bench_half.py
 python experiments/sliding_boundary/compare.py --scan initial
 python experiments/sliding_boundary/compare.py --scan nearby
 python experiments/sliding_boundary/compare.py --scan confirm
 python experiments/sliding_boundary/compare.py --scan edge
 python experiments/sliding_boundary/compare.py --scan structural --variant pruned
 python experiments/sliding_boundary/compare.py --scan structural --variant half
 python experiments/sliding_boundary/fresh.py
 python experiments/sliding_boundary/fresh_half.py
 python experiments/sliding_boundary/fresh_base.py
 python experiments/sliding_boundary/validate_boundary.py
 python experiments/sliding_boundary/bench_boundary.py
 compute-sanitizer --tool {memcheck,synccheck,racecheck} --error-exitcode 1 python experiments/sliding_boundary/sanitize.py
 # Repeat tools on sanitize_half.py. Run GPU commands sequentially.
 python experiments/sliding_boundary/inspect_sass.py
 python experiments/sliding_boundary/audit.py
Build directories and generated binaries are excluded from the archive. base.py
compiles the exact original CUDA only for static disassembly comparison.
