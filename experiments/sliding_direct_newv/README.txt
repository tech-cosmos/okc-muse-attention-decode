Session 24: native sliding new-token V dependency, 2026-10-24

Result: rejected. Production submission.py is unchanged and does not import
this directory. These are complete attention implementations, not omissions.

The original single-tile native kernel waits for historical cp.async copies,
synchronizes, writes the new token's quantized K/V into raw shared arrays,
synchronizes, converts all V to FP16, then fences/synchronizes before WGMMA.

direct.cu quantizes the new V to the same FP8 bytes, converts those bytes to
FP16 immediately and writes the final swizzled V tile. Other threads skip that
row in raw-V conversion, so their stores cannot overlap it. New K still replaces
the zero-filled K row after the first barrier. The final proxy fence and barrier
publish K and V together. This eliminates the middle barrier and the raw-V
stores for the new row. It adds row predicates and scalar final-V stores.

conditional.cu is a control retaining the original staging but executing the
middle barrier only in the block containing the new token. The predicate is
CTA-uniform. The first and final barriers remain unconditional in both variants.
All keys, cache stores, scales, FP8 query/cache rounding, FP16 PV operands,
FP32 partials and inherited PDL merge are preserved.

Balanced adjacent AB/BA cold-L2 graph pairs, 60 rounds/shape, H100 us:
                  parent / direct       paired delta      conditional delta
B1 80K fixed       4.256 / 4.288             +0.048               0.000
B2 80K agentic     4.960 / 4.992              0.000              -0.064
B3 80K agentic     5.248 / 5.344             +0.112              +0.096
B8 80K agentic     7.264 / 7.504             +0.192              +0.112

Each variant has its own paired control. Nearby short/page64/DP measurements
are in results/native-*.json. No scored-shape benefit justifies promotion or an
experimental H200 request. Removing a barrier is not sufficient for a gain;
these measurements do not isolate barrier latency from the changed instructions.
No further parameter sweep or favorable-seed selection was performed.

Validation, raw pairs, bit-exact partial/cache comparisons and sanitizer reports
are checked by audit.py. Full 16-case tests exercise three native cases and
13 production fallbacks each. The broad experimental native dispatch is not a
proposed production dispatch: in particular its B16/32 sliding path loses.
Full bench aggregates 1.920/1.925x (direct/conditional) are local experimental
measurements, not production regressions or H200 estimates.

Fresh testing: 60 native graph replays per variant, changing QKV, caches,
routes, lengths, positions and cache slots, including empty contexts. Ten cases
cover TP8/DP, pages1/64, scales1 and1/64, short/long, soft/coherent/edge data.
Bit-exact testing compares FP32 partials and new cache bytes against the exact
native baseline across 12 case/seed combinations per variant.
Isolated sanitizer drivers perform independent Torch partial reduction and
reference output/cache checks, with no Triton loader. They run 12 calls per
variant/tool. See audit.json for verified summaries; no profiler timing,
dynamic-instruction or achieved-occupancy claim is made.

The comparison and fresh/sanitizer drivers hold /tmp/okc-gpu.lock through data
generation, reference checks, capture and replay. Each timed graph has a 512 MB
L2 flush and empty-replay subtraction. Full bench retains bench.py's timing-loop
lock and runs sequentially with our other GPU work. EARLY=True has runtime
coverage; EARLY=False is compiled only. Static SASS/resource files cover both.

Reproduce from repository root with the existing ~/okc environment:
  python experiments/sliding_direct_newv/generate.py
  NEWV_VARIANT=direct python experiments/sliding_direct_newv/validate.py
  NEWV_VARIANT=direct python experiments/sliding_direct_newv/compare.py
  NEWV_VARIANT=conditional python experiments/sliding_direct_newv/validate.py
  sh experiments/sliding_direct_newv/run_rest.sh
  python experiments/sliding_direct_newv/audit.py
Redirect the first three GPU commands to their named results files as recorded
in the session notes if rebuilding the audit. run_rest.sh writes its logs.
Sources and drivers derive from the prior fp8pairs experiment; the native base
is checked against current production, and generate.py reproduces both variants.
