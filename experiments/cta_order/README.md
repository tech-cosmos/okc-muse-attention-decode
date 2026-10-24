# Session 30: native sliding CTA traversal order

Rejected as a scored optimization. Production is unchanged. Native launch order
now groups KV heads of the same request/split together: `(S,B,Hkv)` becomes
`(Hkv,S,B)`, with matching block-index reads. This encourages neighboring cache-row
work to launch together; it does not guarantee any particular hardware scheduling.
The question was whether it would reduce repeated memory traffic or latency.
All arithmetic, precision, keys, partial layout, cache writes, PDL behavior and
shape predicates remain unchanged. No Triton/Gluon implementation or plan changes.

Frozen production SHA256:
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`.
Candidate SHA256:
`2829404d0de9410a013ef399cd5db2d74e9f190a31f93a0a173e5e64a2b1ae06`.
`generate.py` reproduces exactly the two indexing/launch edits plus extension name.
The source snapshots are standalone experimental modules, not production imports.

## Complete graph measurements

H100, 512 MB L2 flush before every replay, empty replay subtraction, shared GPU
lock throughout generation/reference/capture/timing. Each contrast uses adjacent
alternating AB/BA graph replays. All graphs compute complete valid outputs.
Sixteen initial cases with 80 pairs/contrast; eight representative confirmations
with 120 pairs/contrast, a second input seed and reversed graph capture order.
Each case compares the candidate and a separately captured identical-source
control against production: 4,480 adjacent pairs in total.

Median candidate-minus-production deltas in microseconds:

| Shape | Initial | Confirmation | Identical-code control, initial/confirmation |
|---|---:|---:|---:|
| B1, 80K, page1 (scored) | 0.000 | -0.096 | +0.080 / -0.032 |
| B2, 80K, page1 | -0.032 | -0.128 | +0.032 / -0.080 |
| B8, 80K, page1 | -0.112 | -0.096 | +0.032 / 0.000 |
| B1, 80K, page64 | +0.048 | 0.000 | 0.000 / -0.192 |
| B8, 80K, page64 | -0.192 | -0.288 | +0.048 / -0.128 |
| DP1, 4096, page1 | +0.032 | -0.128 | +0.160 / -0.096 |
| DP1, 4096, page64, scale1/64 | 0.000 | -0.192 | -0.016 / -0.096 |

Small offsets also occur between identical implementations. Paired deltas do not
necessarily equal differences of individual medians, and this is not an H200 or
aggregate-speedup estimate. Nearby B8 gains are modest; the scored B1 result does
not establish enough benefit to promote the source or consume a shared check.
No new batch/page cutoff was fitted. Full candidate benchmark: 1.980x against
fixed SGLang/H200 numbers, an unpaired experimental measurement only.

## Mechanism evidence

All four compiled page/order specializations retain 74 registers/thread, 17,408
shared bytes (cuobjdump), zero local/stack allocation. Across 4,232 decoded SASS
instruction sites, only 12 coordinate-register reads change (three per kernel).
All remaining decoded instruction text is identical, including synchronization,
loads/stores and attention math. Machine encodings differ at those coordinate
reads; do not describe the full SASS as byte-identical. `audit.py` verifies this.

Six sequential NCU launches ran after all benchmark/fresh work, under the shared
GPU lock. Independent output/cache references pass after each. Whole-kernel read
bytes, production/candidate: B1 page1 553,216/553,472; B8 page1
4,300,032/4,299,776; B8 page64 4,236,288/4,236,544. Instruction counts are identical
within each pair (164,482 / 1,315,856 / 1,336,336 respectively). L2 read-sector counts
vary slightly, without a consistent reduction. These observations do not support
HBM read amplification as a removable bottleneck here. They do not isolate a
particular cache's latency or prove an achieved-occupancy explanation. No profiler
duration is used as a graph time. Raw CSV logs and summary are retained.

## Validation

- Candidate full suite: 16/16, three changed-native and thirteen fallback cases;
  followed by all 12 full benchmark shapes.
- 216 independent output/cache reference checks at graph-comparison checkpoints.
- 64 actual-native fresh graph replays: 16 shapes, four changed inputs each,
  pages1/64, TP8/DP, scales1/1/64, soft/coherent/edge inputs. QKV, caches, routes,
  lengths, positions and slots change; final replays shorten/empty contexts.
- Six independently checked profiler calls, as above.
- No new sanitizer claim: no address formula, buffer shape, synchronization or
  arithmetic changes beyond the bijective CTA mapping. Runtime uses EARLY=True;
  EARLY=False has static SASS/resource coverage only.
- Full suite/bench uses bench.py's normal timing-loop lock; custom drivers lock
  their entire GPU workloads. No own GPU workloads overlap.

Reproduction from the repository root, in the existing venv:

```sh
export TORCH_EXTENSIONS_DIR="$PWD/experiments/cta_order/.build"
python experiments/cta_order/generate.py
python experiments/cta_order/run.py suite
python experiments/cta_order/run.py compare
python experiments/cta_order/run.py compare --indices 0 1 3 4 7 11 12 13 --rounds 120 --seed 30919 --reverse --name confirmation
python experiments/cta_order/run.py fresh --name fresh
python experiments/cta_order/collect_profiles.py
python experiments/cta_order/summarize_profiles.py
python experiments/cta_order/audit.py
```

For archived SASS/resources, run `cuobjdump --dump-sass` and
`--dump-resource-usage` on the parent/candidate extension `.so` files in `.build`.
Raw logs are retained in `results/`; build outputs are ignored. No judge jobs.
