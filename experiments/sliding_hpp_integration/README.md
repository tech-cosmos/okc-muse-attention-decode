# Session 43: integrate main 8c4394f

Retained late integration on top of the full-decode integration c6e9fdf. Main
uses paired KV heads (HPP2), 64-key tiles and four warps for sliding decode with
batch*Hkv>32. No native implementation, native dispatch or FP32 compatibility
guard changes. No independent Triton tuning. The new path is disjoint from
native sliding dispatch (batch*Hkv<=16), so no repeated native-vs-main audit.

Frozen sources are `previous.py` (c6e9fdf), `upstream.py` (main 8c4394f), and
`integrated.py` (merge 2ce4d90). The rejected native load-order candidate is not
included. No exact-source H200 qualification or judge request.

Complete H100 graph paired integrated-minus-previous deltas, microseconds;
80 adjacent AB/BA pairs per case. Confirmation changes seed and reverses capture:

| Sliding L1 case | initial | confirmation |
|---|---:|---:|
| B32 TP8 80K page1, scored | -0.576 | -0.416 |
| B24 TP8 80K page1 | -1.008 | -0.736 |
| B48 TP8 80K page1 | -1.936 | -1.920 |
| B32 TP8 80K page64, scaled edges | +0.480 | +0.448 |
| B3 DP 6K page64, scaled edges | -0.016 | +0.704 |
| B17 TP8 5K page1, edges | -0.096 | +0.064 |

Four unchanged initial controls range -0.064..+0.032us; B1 full decode is
exactly 0us paired. The page64 B32 regression repeats; DP is mixed. Report these
to the main owner, without fitting a page/head/context cutoff. Retain upstream's
scored and larger page1 improvements. No claim of uniform nearby improvement.

Validation: 16/16 standard tests, then all 12 benchmark shapes. The standard
suite does not exercise this new large sliding-decode plan; six explicit cases
provide changed-path coverage and four are controls. Ten initial + six
confirmation cases yield 1280 adjacent pairs and 96 independent output/cache
reference checks. Eighty fresh graph replays (10 cases x 4 inputs x 2 sources)
change QKV, cache, routes, lengths, positions and cache slots, including halved
and empty histories. TP8/DP, page1/64, scales1/1/64 all pass.

Final standard bench aggregate 2.006x versus fixed SGLang/H200 numbers is
unpaired local evidence, not a regression estimate versus the earlier 2.035x
bench: other unchanged shapes varied between runs. Paired graphs isolate this
integration's effects. Separate medians need not subtract to paired deltas.

`audit.py` checks 16 shared top-level function/class ASTs against main (including
nested helper bodies), unchanged native definitions/dispatcher, exact hashes,
raw medians and all counts. No new sanitizer or profiling claim. Custom drivers
hold the shared GPU lock through generation/reference/capture/replay, flush
512MB before measurement and subtract an empty replay. Standard suite holds the
lock; standard bench uses its timing-loop lock. Own GPU work ran sequentially.

Reproduce with the repository venv, redirecting modes to their matching `.log`:

```sh
python experiments/sliding_hpp_integration/validate.py suite
python experiments/sliding_hpp_integration/validate.py compare
python experiments/sliding_hpp_integration/validate.py confirm
python experiments/sliding_hpp_integration/validate.py fresh
python experiments/sliding_hpp_integration/audit.py
```
