Codex session 3: native gathered full-attention experiments (H100, 2026-10-23)

No code here is imported by submission.py, and none was sent to the judge.
Production remains the session-2 native small-sliding merge plus main's Triton.
This directory is retained to reproduce a negative result, not as a new dispatch.

Recommended entry points
------------------------
  . /home/daytona/okc/bin/activate
  python experiments/gather/compare_final.py
  NATIVE_MODULE=pair_mbar python experiments/gather/validate.py
  NATIVE_MODULE=pair_mbar ONLY_GRAPHS=1 CHANGE_LENGTHS=1 python experiments/gather/validate.py
  NATIVE_MODULE=pair_mbar compute-sanitizer --tool memcheck --error-exitcode 1 python experiments/gather/validate.py sanitize
  NATIVE_MODULE=pair_mbar compute-sanitizer --tool synccheck --error-exitcode 1 python experiments/gather/validate.py sanitize
  NATIVE_MODULE=pair_mbar compute-sanitizer --tool racecheck --error-exitcode 1 python experiments/gather/validate.py sanitize
  python experiments/gather/validate_gather.py
  GATHER_MODULE=gather_bulk python experiments/gather/validate_gather.py

These use load_inline, CUDA 13, sm_90a. Build products stay in ignored .build-*
directories. Sanitizer binaries are in /usr/local/cuda-13.0/bin. The validation
scripts hold /tmp/okc-gpu.lock; all measurements use tools/bench.time_graph's GPU
lock, cold L2, median CUDA graph replay and empty-graph subtraction. The final
comparison alternates order across three paired measurements. Do not use eager
kernel timing as a substitute. Some exploratory minima differ by several percent
from final medians; quote the final comparison when reporting the retained kernel.

The retained attention prototype, pair_mbar.py
--------------------------------------------
Two producer warps gather both KV heads in contiguous 128-byte global rows. Two
consumer warpgroups (four warps per head) perform FP8 WGMMA QK and FP16 warp MMA PV,
with FP32 accumulation. It computes query normalization, new FP8 cache writes,
causal new-token attention, and every cached key inside the call. The existing
Triton _combine reduces FP32 partials and applies the output norm/gate.

Each stage has a full and an empty Hopper mbarrier. The full barrier counts the
64 producer threads; cp.async.mbarrier.arrive.noinc publishes completed gathers.
New-token shared stores use release arrivals after producer synchronization and
an async-proxy fence. Each consumer group arrives once at the empty barrier after
its reads finish. Phase is (tile // stages) & 1. Producer reuse waits for the
previous empty phase. This removes software polling and early wait_group gating.
This is a real improvement over the flags version, but still too slow overall.

Supported: full attention, L=8, D=64, G=8, an even number of KV heads, pages 1/64.
Use stages 2, 3, 4 or 5; the final selected configuration is stages=4 and
splits=max(1,264 // (batch * num_kv_heads // 2)). Other prototype wrappers retain
historical stage aliases; reproduce the plans in their JSON files, not arbitrary
stage arguments. No attention kernel is intended to be imported into production.

Final H100 paired medians, full verify at context 80000
-----------------------------------------------------
 batch     production       native pair_mbar
   1          18.720 us          34.864 us
   8          88.080 us         127.216 us
  16         158.064 us         231.296 us
  24         226.944 us         330.896 us
  32         307.264 us         447.072 us

Data/results: results/final-comparison.json and .log. Production's complete
12-shape bench was 1.827x against the fixed SGLang/H200 values in tools/bench.py.
This is not an H200 judge measurement, and production code did not change.

What was learned
----------------
* Pure native gather, with exact uint32 byte checksums, can reach 128/246 us at
  batches 16/32. Cooperative 4-warp copies, 2 heads, 3 stages, 1056 nominal CTAs
  won the initial sweep. One producer warp was ~156/308 us at the same grid.
* Two producer warps approach 258 us at batch 32 with 1584 nominal CTAs. Four
  producer warps lose residency and bottom out around 299 us. This does not carry
  over to attention: attention's registers/shared memory allow far fewer CTAs.
* Software-flag paired attention: 1/2/4 producer warps best ~563/497/544 us at B32.
  Cooperative 8-warp loads are ~595 us. The native single-group dual-chain form
  is ~497 us; adding warpgroup PV, page broadcasts, or cooperative loads loses.
* Hardware barriers improve paired attention to ~438 us in the exploratory sweep;
  the final three-pair median is 447 us. Single-group dual with hardware barriers
  remains ~495 us. This is no improvement over session-1 native throughput.
* Bulk asynchronous 128B row gathers reach ~260 us without attention. Tensor-map
  row loads can feed a SW128 WGMMA layout directly and are correct on scored
  comparisons, but full attention is ~841 us. Linear bulk K/V plus an explicit
  shared transpose is ~872 us. Small random rows do not amortize the copy-engine
  cost at the attention kernel's limited resident CTA count.
* NCU on the 4-producer paired variant: 80 regs/thread, 52 KB shared/CTA, 2 CTAs/SM,
  35.7% achieved occupancy, 32.4% DRAM utilization, 0.53 issued warps/scheduler.
  Long scoreboard, short scoreboard, and execution-dependency wait each cost
  about 2.1 warp cycles per issued instruction; barriers ~0.86. A full-row load
  alone does not fix synchronization/instruction/resource overhead.

Validation
----------
* Flags prototype: 16/16 native/fallback reference tests, plus 35 graph replays
  with fresh QKV/cache/page tables on seven nearby shapes.
* Retained mbarrier prototype: 16/16 reference suite; the same 35 fresh-data
  replays; another 35 replays with changed sequence lengths, positions, cache
  locations and FA3 lengths, including empty contexts. All pass.
* mbarrier memcheck/synccheck: 0 errors. Racecheck: 0 hazards, errors or warnings.
  Four small native cases (page 1/64, scales 1 and 1/64), three calls per case.
* Direct-copy and bulk-copy gather outputs each match independent PyTorch
  checksums for 128 combinations of page 1/64, TP8/DP heads, gather head counts,
  producer counts and depths. Earlier gather-validation.log said '64 variants'
  per shape; it actually ran 32, corrected in the script and bulk log.
* Every attention sweep checked finite outputs and relative difference <2.5%
  against a reference-validated production output. Observed differences ~0.03%
  or less. Only pair_p2 and pair_mbar received the full extra validation above.
* Production independently passed 16/16. Later main commits changed only notes
  and judge profiles; the production kernel is byte-for-byte unchanged.

Source organization
-------------------
  gather_v1.py             initial 0/1-producer checksum kernels
  gather.py                0/1/2/4-producer checksum kernels
  gather_bulk.py           linear bulk-copy checksum kernels
  pair.py, pair_v1.py      first paired-head attention, one producer warp
  pair_p2.py, pair_p4.py   two/four producer warps
  pair_coop.py             cooperative loading, two consumer groups
  dual*.py                one consumer group with two softmax chains
  pair_mbar.py            retained hardware-barrier prototype
  pair_tma.py             tensor-map K with SW128 layout, linear bulk V
  pair_bulk.py            linear bulk K/V, explicit shared K rearrangement
  single_head.py, wgmma_pv_reference.py   session-1 source used for transformations
  make_*.py               historical source transformation recipes
  bench_*.py              historical sweeps; results are keyed by those filenames
  profile_pair.py         ncu reproducer for pair_p4 (holds GPU lock)

NVIDIA references used to check barrier ordering and descriptor syntax:
https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html
Also read CUDA 13's installed cccl/cuda/__ptx/instructions/generated headers
(cp_async_mbarrier_arrive_noinc, mbarrier_try_wait_parity, mbarrier_arrive,
cp_async_bulk). The native experiment implements the instructions itself.
