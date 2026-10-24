# Session 43: integrate main 9a43323

Retained integration; no new native implementation or Triton tuning. Main now
uses HPP2, block_n64, stages5 for batch*Hkv<=2 full decode. Both native extension
definitions, `kernel_fn`, native dispatch predicates and the FP32 compatibility
guard remain identical to the previous source. Main's smaller tile changes
actual split counts, so nearby native-dispatch boundaries are explicitly tested.
The rejected load-order experiment is on codex-exp/native-merge-load-issue and
is not imported or merged here.

Frozen sources: `previous.py` = Codex d1d801e, `upstream.py` = main 9a43323,
`integrated.py` = merge c6e9fdf. Final submission SHA256:
`16b6aa31ac9816b65177b08b68948fcc6d711cdc2169f4144c02fe874a9649c4`.
This exact source has no H200 qualification.

H100 complete CUDA graphs, 80 adjacent AB/BA pairs per contrast. Negative paired
deltas mean integrated is faster; microseconds, initial/second-seed confirmation
with reversed capture order:

| Full L1 TP8 case | integrated minus previous | integrated minus upstream |
|---|---:|---:|
| 80K page1, scored | -0.448 / -0.688 | -0.560 / -0.576 |
| 80K page64 | +0.160 / +0.032 | -0.416 / -0.416 |
| 4095 page1, scaled soft | +0.288 / +0.192 | +0.032 / -0.096 |
| 4096 page1, scaled soft | -0.288 / -0.384 | -0.384 / -0.512 |
| 8192 page64, scaled edges | +0.992 / +0.992 | -0.384 / -0.704 |
| 33333 page64, scaled soft | -0.368 / -0.896 | -0.352 / -0.624 |
| 120K page1 | -0.016 / -0.128 | -0.144 / -0.416 |

The 8K regression against old production repeats, while native still improves
the new upstream on that case. Report it to the Triton owner; no fitted context
or page cutoff. 4095 has 64 splits and uses the unchanged upstream merge;
4096 has 65 splits and now reaches native full (previously 33, non-native).
Native comparison includes its FP32 producer storage versus upstream BF16;
it does not isolate the merger's arithmetic. Other initial short cases:
ctx10/page1 +0.064us and ctx1000/page64 +0.448us versus old production. Four
unchanged-path controls range -0.176..+0.096us. Separate median times need not
subtract to paired deltas; small effects should not be overinterpreted.

Validation:

- Standard suite 16/16, then all 12 benchmark shapes. The new B1 full L1 plan
  is absent from that suite; the explicit cases supply the changed-path coverage.
  Local aggregate 2.035x versus fixed SGLang/H200 times is descriptive only.
- 13 initial + 7 confirmation cases, two contrasts each: 3200 adjacent pairs,
  180 independent output/cache checks. Nine initial cases reach the changed
  upstream plan; four are unchanged controls.
- 156 fresh graph replays: 13 cases x 4 inputs x 3 sources. Fresh QKV, FP8 cache,
  routes, lengths, positions and cache slots, including halved/empty histories,
  pages1/64, scales1/1/64, soft/coherent/edges, and a DP control. All pass.
- `audit.py` reproduces raw medians/counts, verifies 16 top-level shared
  function/class ASTs against main (nested helpers are included in their parent
  AST), unchanged native definitions and dispatcher, and source hashes.

Custom drivers hold the shared GPU lock over generation/reference/capture/
replay, flush 512MB before each measured graph and subtract an empty replay.
Suite holds the lock; standard bench uses its timing-loop lock. All own work
sequential and finished. No new sanitizer, profiling or H200 claim.

Reproduce in the repository venv, redirecting each mode to its `.log`:

```sh
python experiments/d1_full_integration/validate.py suite
python experiments/d1_full_integration/validate.py compare
python experiments/d1_full_integration/validate.py confirm
python experiments/d1_full_integration/validate.py fresh
python experiments/d1_full_integration/audit.py
```

No check/scored requests. Codex's daily scored share is exhausted; preserve the
remaining shared check. Check this exact source before any future scored run.
