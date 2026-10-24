# Session 29: native full-decode neighborhood and merge composition

No production change or judge request. This is a local H100 investigation of
the incidental short-context wins from session 28, not a scored optimization.
The frozen production SHA256 is
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.

## Frozen attention, broad measurements

`tile128.cu`, `binding.cpp`, `candidate.py`, and `production_snapshot.py` are
byte-identical to experiment `081f126`. No CUDA attention instruction, native
merge kernel, Triton/Gluon source, or plan was tuned. The wrapper computes all
full-attention L1 keys, writes the new FP8 cache entry, and uses FP32 partials.

The initial matrix has 74 cases: batches 1/2/4/8, contexts
128/1024/4096/9000/16000/32768, both pages; matched 9K scale/distribution controls;
tile boundaries, DP, and 80K controls. Each has 60 adjacent AB/BA cold-L2 graph
pairs. Twenty-two representative cases have 100 second-seed pairs each.

Original tile wrapper minus production, paired median microseconds:

| batch/context/page | initial | second seed |
|---|---:|---:|
| 1 / 4096 / 1 | -0.480 | -0.352 |
| 1 / 9000 / 1 | +0.208 | +0.224 |
| 1 / 9000 / 64 | 0.000 | +0.160 |
| 2 / 4096 / 1 | -1.824 | -2.048 |
| 2 / 9000 / 1 | -2.000 | -2.240 |
| 2 / 9000 / 64 | -2.864 | -2.912 |
| 2 / 16000 / 1 | +0.048 | -0.672 |
| 4 / 9000 / 1 | +0.352 | +0.016 |
| 8 / 4096 / 1 | +0.816 | +0.192 |
| 8 / 4096 / 64 | +0.160 | +0.048 |

The B2/9K gain repeats across both pages, scales 1 and 1/64, and soft/coherent
distributions: initial deltas -3.056 to -2.000 us; confirmation -3.040 to -2.192 us.
It is not limited to the single scaled/page64 observation. Gains do not extend
uniformly across batches or contexts. Both DP/4096 controls lose, +1.248/+0.848 us.
The B1/80K scored control still loses 6.448 us; B2/80K page64 loses 10.448 us.
No fitted context cutoff was promoted.

## Existing native merge as a control

The parent already dispatches its native full merge when `B*Hkv <= 4` and
`64 < splits <= 128`. The frozen tile wrapper used Triton's merge throughout.
`native_merge_control.py` applies exactly the parent's native merge predicate
to the tile wrapper's actual split count. It changes no attention or merge code.

Sixteen shapes (B1/2, contexts 4096/8192/9000/16000, pages1/64) each compare the
composition with both parent and original tile using 80 adjacent pairs per
contrast. A second seed reverses graph capture order and repeats all contrasts.
Ten shapes actually select the native merge; six are identical-code controls.

Composition minus original tile, initial / confirmation, microseconds:

| batch/context/page | initial | reversed capture, second seed |
|---|---:|---:|
| 1 / 8192 / 1 | -0.656 | -0.640 |
| 1 / 9000 / 1 | -0.544 | -0.496 |
| 1 / 9000 / 64 | -0.576 | -0.624 |
| 1 / 16000 / 1 | -0.640 | -0.768 |
| 2 / 8192 / 1 | -0.576 | -0.416 |
| 2 / 9000 / 1 | -0.560 | -0.576 |
| 2 / 9000 / 64 | -0.512 | -0.608 |

The composition removes the original B1/9K regression: paired deltas against
production are -0.272/-0.464 us on page1 and -0.208/-0.528 us on page64.
Initial identical-code controls show +0.016 to +0.160 us offsets; with reversed
capture order and a new seed they are -0.064 to +0.016 us. Allocation/capture
effects limit fine-grained timing interpretation. The merge gain repeats well
beyond those offsets, but this is not an isolated hardware latency measurement.

The composition does not change the losing 80K attention topology and is not a
proposed production dispatch. Its complete broad-wrapper benchmark is 1.622x
against fixed SGLang/H200 reference numbers, not a production regression or
H200 estimate. Short-context improvements alone do not justify a scarce check.

## Validation and reproduction

- 11,760 adjacent graph pairs; 864 independent output/cache checks across
  comparison checkpoints (before capture, after replay, after timing).
- 64 fresh graph replays of the original tile and 64 of the composition,
  changing QKV, cache values, routes, lengths, positions, and cache slots,
  including shortened and empty contexts. Of the composition replays, 40 use
  the native merge and 24 use the inherited Triton merge.
- Composed wrapper: full 16/16 suite, then all 12 benchmark shapes. Five suite
  cases use native attention; one also exercises the composed native merge.
- No new sanitizer run: both CUDA kernels are unchanged. Session28's sanitizer
  evidence applies to its own tested cases, not new coverage claimed here.
- Production source is unchanged; no redundant production suite/bench was run.

Custom drivers lock input generation, reference checks, capture, and timing.
Each timed replay follows a 512MB L2 flush; an empty replay is subtracted.
The suite/bench driver keeps bench.py's timing-loop lock, with no overlapping
GPU process launched by this session. Paired medians need not equal differences
of separate medians. Bootstrap intervals in `summary.json` describe within-run
variation only, not H200 uncertainty or simultaneous/multiple-comparison bounds.

From the repo root, activate `/home/daytona/okc/bin/activate`:

```sh
python experiments/full_decode_neighborhood/run.py compare --rounds 60 --name initial
python experiments/full_decode_neighborhood/run.py compare --rounds 100 --seed 29207 --name confirmation --indices 4 5 6 7 16 17 18 19 20 21 28 29 30 31 40 41 54 55 56 57 58 59
python experiments/full_decode_neighborhood/run.py fresh --seed 29301 --name fresh --indices 0 1 14 15 16 17 18 19 30 31 40 41 55 58 70 71
python experiments/full_decode_neighborhood/merge_control.py suite
python experiments/full_decode_neighborhood/merge_control.py compare
python experiments/full_decode_neighborhood/merge_control.py fresh
python experiments/full_decode_neighborhood/merge_confirmation.py compare
python experiments/full_decode_neighborhood/audit.py
```

Redirect output to matching `results/*.log` files (suite is `merge-suite.log`,
fresh composition is `merge-fresh.log`). The audit checks frozen source hashes,
the precise wrapper AST difference, all sample and validation counts, and
recomputes paired deltas. Source manifests, raw pairs, and summaries are archived.
Only width128 is included; the inherited wrapper retains its old default256,
so explicitly pass128 when calling it outside these drivers.
