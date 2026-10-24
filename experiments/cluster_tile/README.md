# Session 31: single-tile sliding cluster fusion

Rejected local experiment. Production `submission.py` is unchanged; nothing in
this directory is imported by production. No judge jobs were requested.

The distinct mechanism is a portable eight-CTA Hopper cluster over the full
2048-token window. Each CTA computes one 256-key tile with no streaming loop.
After WGMMA completes, it reuses tile shared storage for FP32 partials. A cluster
barrier publishes those partials; each CTA's first warp reduces one query head
through distributed shared memory, then performs RMSNorm/gate/BF16 output. A
second cluster barrier prevents exit before all remote reads finish. No atomics,
completion counters, global partial writes or separate merge launch are needed.
All keys and values are attended; query/cache precision is inherited unchanged.
The host still allocates unused partial tensors to preserve a common binding.

The earlier session2 cluster experiment used slower streaming attention. This
experiment tests fusion with the newer direct single-tile implementation. The
unfused control uses the identical 256-key attention body, eight global partials
and the unchanged production Triton merge. No Triton code or plan was tuned.
Both wrappers apply broadly to supported sliding L1 shapes for this experiment;
this is not a proposed production dispatch.

## Results

H100, complete CUDA graphs, 512 MB L2 flush before each replay, empty replay
subtraction, 60 adjacent alternating AB/BA pairs for each contrast. All custom
GPU work holds `/tmp/okc-gpu.lock`; suite/bench uses bench.py's timing-loop lock.

| sliding L1 | production us | cluster us | paired cluster delta | unfused us | paired unfused delta |
|---|---:|---:|---:|---:|---:|
| B1 80K page1 | 4.320 | 7.232 | +2.928 | 6.528 | +2.240 |
| B2 80K page1 | 4.992 | 7.488 | +2.496 | 7.136 | +2.208 |
| B8 80K page1 | 7.392 | 9.936 | +2.608 | 9.408 | +2.048 |
| B16 80K page1 | 10.240 | 18.816 | +8.512 | 12.928 | +2.672 |
| B32 80K page1 | 15.712 | 31.072 | +15.312 | 23.536 | +7.968 |
| B1 80K page64 | 4.320 | 7.216 | +2.848 | 6.400 | +2.032 |
| DP1 4096 page64 scale1/64 | 7.344 | 9.952 | +2.640 | 8.912 | +1.472 |

Each contrast has its own paired production times; the table shows production
from the cluster contrast. Deltas are medians of paired differences, not
necessarily differences of medians. All 18 shapes lose in both contrasts. The
losses are decisive; no seed selection, confirmation sweep or cutoff fitting.
Full broad-wrapper benches: cluster 1.826x, unfused 1.883x against fixed
SGLang/H200 times. These are experimental H100 results, not production
regressions or H200 speedup estimates.

Fusion does not rescue the 256-key tile's latency and adds more cost at large
batch. Resource reports: cluster236/unfused223 registers, each65536 dynamic plus
1024 static shared bytes; zero stack/local allocation. SASS has real cluster
arrive/wait instructions. These observations do not isolate synchronization
cost from scheduling/residency or compiler effects. No NCU/achieved occupancy
claim. Close this fixed-window cluster topology; do not follow with a cluster
size/tile-width micro-sweep.

## Validation and reproduction

- Both wrappers pass16/16 reference tests, each3 actual-native/13 fallback cases.
- Both complete all12 benchmark shapes after correctness. Cluster's first suite
  log also retains a harness-only AttributeError: bench required `_plan` on the
  wrapper. The hook was exported without changing GPU code, then the full bench
  completed in `bench-cluster.log`; the16 correctness passes precede that error.
- 2160 adjacent graph pairs;162 independent output/cache reference checks around
  capture/timing on18 shapes. TP8/DP, pages1/64, scales1/1/64, soft/coherent/edges.
- 144 actual-native fresh graph replays (72 per wrapper): changed QKV, caches,
  page tables, sequence lengths, positions and new cache slots; halved and empty
  contexts included. Captures are reused with input copies.
- Isolated fused-native sanitizer:24 fresh replays per tool, pages1/64,
  scales1/1/64, contexts10/257/4096 and empty contexts. No Triton kernels launched.
  Memcheck/synccheck:0 errors; racecheck:0 hazards/errors/warnings. Coverage applies to
  the fused kernel, not an independent unfused sanitizer run.
- CPU audit regenerates sources, checks exact storage/gather/conversion coverage,
  preserved production identity, test counts, paired medians and sanitizer logs.

Activate `~/okc`, then run `python experiments/cluster_tile/build_generator.py`,
`generate.py`, `fuse.py` (each with its full directory prefix). `run.py suite
--variant cluster|unfused`, `run.py compare`, and `run.py fresh --variant ...`
provide the GPU checks. Sanitizer path: `/usr/local/cuda/bin/compute-sanitizer`.
`python experiments/cluster_tile/audit.py` is CPU-only. `.build` is ignored.
The generator is derived from session28's fixed generator; it preserves sliding
RoPE and window bounds. Frozen `.cu`, binding and production snapshots are saved.
