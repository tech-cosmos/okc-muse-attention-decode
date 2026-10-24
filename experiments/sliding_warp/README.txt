Session 17: single-tile, warp-sized native sliding attention
=========================================================

These are experiments, not production imports. submission.py remains byte-for-byte
source 3fb391f (SHA256 16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e).

Hypothesis
----------
L1/G8 attention uses eight real rows in a 64-row WGMMA tile. A warp-sized
m16n8 tensor-core operation could reduce padded work and CTA resources. This
is a new single-tile/control implementation, not the old session2 multistage
warp-loop design or session16's padded-row softmax-only pruning.

base.cu is the exact embedded production native attention source. generate.py
preserves its early cp.async gathers, query normalization/RoPE/FP8 rounding,
cache insertion, 64-key splits and FP32 partial format. P/V operands are FP16.
Every historical key and the new token contribute; masked slots are only outside
the causal window. The production Triton PDL merge is unchanged.

warp: one 32-thread CTA gathers the full tile and runs m16n8k32 FP8 QK,
then m16n8k16 FP16 PV, with 16 rows (eight real, eight padded).
cooperative: the same warp-zero math but 128 threads cooperate on gathers and
value conversion. It separates reduced gather parallelism from matrix changes.
hybrid: keep native m64n64k32 FP8 WGMMA QK, then warp-zero softmax/m16n8k16 PV.
It isolates the smaller PV without the FP8-to-FP16 QK lowering observed in SASS.
The FP16 value staging is linear for ldmatrix.trans in all three experiments.

candidate.py dispatches broadly to L1 sliding G8/D64/page1-or64 for experimental
coverage; other shapes use unchanged production. This broad dispatch is NOT a
proposed production patch. It does not include session16's boundary patch.

Reproduce from repo root, in ~/okc, with CUDA_HOME=/usr/local/cuda-13.0:
  python experiments/sliding_warp/generate.py
  WARP_VARIANT=warp python experiments/sliding_warp/validate.py
  WARP_VARIANT=warp python experiments/sliding_warp/bench.py
  WARP_VARIANT=warp python experiments/sliding_warp/compare.py --control native
  WARP_VARIANT=warp python experiments/sliding_warp/fresh.py
Repeat with cooperative and hybrid. Run benchmarks sequentially, with no other
local benchmark/profile work; custom drivers hold /tmp/okc-gpu.lock throughout
reference checks, capture and timing. bench.py retains tools/bench.py's timing
lock. Every timing flushes 512MB before each graph replay and subtracts an empty
replay. compare.py uses 60 adjacent AB/BA pairs per shape, with output/cache
reference checks before capture, after replay and after timing. Raw pairs are
in results/native-*.json. Per-shape medians need not subtract to paired deltas.

Native-only sanitizers (no Triton loader):
  WARP_VARIANT=warp /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck \
    --error-exitcode 99 python experiments/sliding_warp/sanitize.py
Repeat tools synccheck/racecheck and all three variants. Each invocation runs
12 native calls (two pages, two scales, three changing-QKV calls) and independently
reduces FP32 partials in torch before checking output and new cache entries.
All runtime tests select EARLY=True; EARLY=False is compiled but unused here.

  python experiments/sliding_warp/inspect_sass.py
  python experiments/sliding_warp/audit.py
inspect_sass reads the compiled production extension as its baseline, then the
local experiment builds. It records static instruction sites and resources,
not executed instruction counts, achieved occupancy, or graph latencies.
audit reproduces generated sources and checks all reported completed evidence.

Layout controls
---------------
The initial three variants used linear V shared memory. To separate matrix choice
from that avoidable layout change, generate.py also emits warp_swizzle,
cooperative_swizzle and hybrid_swizzle. They preserve the parent's SW128-like
half-index XOR mapping, and apply the same mapping to ldmatrix row addresses.
Use those names in WARP_VARIANT for tests, full benches, paired comparisons and
fresh replays. These controls have no new sanitizer claim: native mem/sync/race
reports apply to warp, cooperative and hybrid only. All control timing is fully
serial with the earlier timing and sanitizer runs. See final results below.

Outcome (H100, cold-L2 graphs; microseconds)
------------------------------------------
No production change. All variants lose the batch1 scored target.
Each row is an independent balanced comparison against original native.

variant                 B1 parent/candidate (delta)   B2 parent/candidate (delta)
warp                    4.480/6.432 (+1.920)  5.120/6.752 (+1.664)
cooperative             4.448/5.472 (+1.056)  5.072/6.096 (+1.024)
hybrid                  4.448/4.944 (+0.480)  5.040/5.552 (+0.496)
warp_swizzle            4.608/6.384 (+1.744)  4.912/6.496 (+1.552)
cooperative_swizzle     4.448/5.360 (+0.992)  5.088/5.952 (+0.864)
hybrid_swizzle          4.640/5.088 (+0.288)  5.056/5.600 (+0.576)

The swizzled hybrid has incidental nearby B8/DP1 improvements but loses B1/B2;
no shape rule is promoted. For B16/B32 the original forced-native control is
itself slower than production, so parity there is insufficient.

Warp/cooperative QK FP8 mma.sync lowers to FP16 HMMA on this Hopper build.
Both have 64 HMMA and 32 LDSM instruction sites versus baseline 2 QGMMA/4 HGMMA.
Hybrid retains 2 QGMMA but replaces 4 HGMMA with 32 HMMA plus 32 LDSM.
The swizzle controls reduce the value-layout disadvantage without recovering
B1 latency. These are static instruction counts; there is no NCU/achieved
occupancy or dynamic instruction claim. More sites and changed gather topology
are consistent with the regressions, not an isolated causal proof.

Validation: each of six variants passes16/16 (3 native/13 fallback), followed
by its full benchmark; each passes60 all-native fresh graph replays (360 total).
There are120 comparison entries across six drivers,60 adjacent pairs per shape,
including controls. Original warp/cooperative/hybrid each pass native-only
memcheck/synccheck/racecheck with zero errors/hazards;12 calls per tool/variant.
Swizzled controls have no separate sanitizer coverage. EARLY=True only is used.
No experiment was sent to the judge. The production source remains unchanged.
All reported logs are complete; audit.json checks source reproduction and evidence.
