# Session 39: native full-merge phase diagnosis

This is an offline diagnostic, not a speedup candidate. Production remains
exact SHA256 `ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca`,
with main06b7755 integrated and the native full-merge FP32 allocation guard.
No Triton helper, plan, native dispatch, attention arithmetic or production
source changed. No judge request was made.

The native small-full merge has exhausted several reduction/epilogue ideas.
This cycle establishes instruction windows before selecting another change.
`generate.py` takes the frozen `previous.py` and changes only its native merge
extension and an appended diagnostic adapter. Each phase specialization reads
clock64 twice and saves each warp leader's stamps in fresh disjoint output.
All attention, cache updates, normalization/gate and real output remain inside
the complete timed wrapper. The adapter's retained trace is for offline reading
only. It does not cache anything for attention or belong in a submitted kernel.

There are seven adjacent source intervals, a whole-body interval, consecutive
clock control and compiled-out no-clock control. Two complete production
captures provide another control. Warps 1–7 exit after publication; their absent
late-interval end stamps are explicitly zero and excluded from summaries.

## Findings and limits

Median warp-zero cycles, initial / second-seed confirmation. Confirmation
reverses module/capture order. Do not sum rows or convert these to a production
latency partition: separate probes change scheduling and register allocation.

| Emitted interval | B1 L1 80K page1 | B1 L8 80K page1 |
|---|---:|---:|
| Prologue | 1291 / 1330.5 | 1597 / 1613 |
| Dependency wait | 27546 / 27206 | 27848 / 28154 |
| Local statistics and weights | 837 / 858 | 989 / 1005 |
| Partial loads and weighted sums | 975 / 981.5 | 1443 / 1432.5 |
| Shared publication and barrier | 61 / 61 | 61 / 61 |
| Cross-warp merge | 376.5 / 372 | 402.5 / 382 |
| Normalize, gate, output store | 257 / 257 | 257 / 257 |
| Whole body | 30962.5 / 30895.5 | 32381 / 32411 |
| Consecutive clocks | 2 / 2 | 2 / 2 |

The wait probe brackets exactly one ACQBULK instruction. It shows substantial
waiting in these probes, not avoidable arithmetic or permission to remove PDL.
Session27 already showed that serializing the merge loses. The epilogue and
publication windows are small relative to the other post-wait windows; this
supports keeping those rejected instruction-level families closed. Local stats
and partial retrieval remain possible areas to investigate, but these probes
establish neither a removable cost nor a concrete winning optimization.

Nearby B2 L8 9K scaled-edge wait probes differ: page1 4881/5260 cycles and
page64 4255.5/4163.5. They do not reproduce or quantify an isolated occupancy
cause from session27; they use different inputs and instrumented binaries.

SASS checks are crucial here. Both no-clock specializations are instruction-
identical to production. All17 coherent load sites remain after ACQBULK.
However, some probes move that wait substantially earlier relative to prologue
instructions. The stats probe raises registers48->61, the partial-load probe
to52, and several controls to58; the other probes stay48. All use3200 reported
shared bytes, zero stack/local. Thus neither summing phase medians nor subtracting
clock-control graph cost gives a valid production decomposition. Consecutive
clocks have no intervening instruction but do not model all perturbation.

Whole-body probe graph deltas for B1 L1/L8 page1 are +0.160/+0.064 and
-0.096/+0.288 us, initial/confirmation. Identical production capture deltas range
-0.192 to+0.112 us; SASS-identical no-clock deltas range-0.128 to+0.176 us.
Stats instrumentation costs up to1.200 us and consecutive-clock control up to
0.928 us nearby, despite the two-clock interval being2cycles. These differences
are a warning against subtractive attribution, not speedups or fitted cutoffs.

## Validation and reproduction

- Whole-body wrapper16/16; only2 standard cases use the modified native merge,
  14 are unchanged controls. Then all12bench shapes, local aggregate1.994x versus
  fixed SGLang/H200 numbers; unpaired diagnostic evidence, not an H200 estimate.
- 6 initial+4confirmation cases,11contrasts each,60adjacent AB/BA complete graph
  pairs:6600pairs,660independent output/cache checks at three checkpoints.
- 1080 separate cold trace replays. Raw timestamps and timing pairs are retained
  in `pairs-*.json.gz`; smaller `summary-*.json` files omit the raw arrays.
- 240fresh native graph replays across6cases and10variants:216instrumented and
  24compiled-out controls. Fresh QKV/cache/routes/lengths/positions/cache slots,
  halved and empty contexts, pages1/64 and scales1/1/64. All independent output/
  cache references pass. Non-PDL specializations have assembly coverage only.
- CPU audit checks source reproduction, unchanged ASTs, exact production hash,
  trace index injectivity/bounds, all counts and raw medians, clock placement and
  SASS identity. No sanitizer or NCU run, and no such coverage is claimed.

Custom drivers lock generation/reference/capture/replay; every timed graph is
preceded by a512MBflush, and empty replay cost is subtracted. Full suite holds
that lock; standard bench retains its timing-loop lock. All own GPU work runs
sequentially. Only generated disassembly/report whitespace was normalized.

```sh
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_EXTENSIONS_DIR="$PWD/.build/session39"
export MAX_JOBS=4
/home/daytona/okc/bin/python experiments/merge_phases/validate.py suite
/home/daytona/okc/bin/python experiments/merge_phases/validate.py compare
/home/daytona/okc/bin/python experiments/merge_phases/validate.py confirm
/home/daytona/okc/bin/python experiments/merge_phases/validate.py fresh
python experiments/merge_phases/audit.py
```

The assembly artifacts were generated with CUDA13cuobjdump `--dump-sass` and
`--dump-resource-usage` on the two merge extensions in that build directory.
No clocks, trace storage or experimental imports were added to production.
