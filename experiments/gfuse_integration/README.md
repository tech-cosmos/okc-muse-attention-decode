# Session 42: integrate upstream fused Gluon merge with native dispatch

This validates origin/main eef9721 (kernel 3bb3a64) integrated with the existing
Codex native kernels. The merge commit is c12bd0f. Native extension definitions,
dispatch and FP32 compatibility guard are unchanged. All 24 shared function/class
ASTs, including nested Gluon helpers, match upstream. This is an upstream
integration, not a new native optimization. No Triton plans were tuned.

Frozen sources: previous.py = codex 18aa622, upstream.py = main eef9721,
merged.py = integrated submission.py. SHA256 hashes are in audit.json.
The fused path uses per-request/head-pair arrival/generation counters; only
one-wave Gluon plans enable it. Other paths keep the separate merger.

Run from the repository root with ~/okc Python, sequentially:

```sh
flock /tmp/okc-gpu.lock ~/okc/bin/python tools/bench.py test
~/okc/bin/python tools/bench.py bench
~/okc/bin/python experiments/gfuse_integration/validate.py compare
~/okc/bin/python experiments/gfuse_integration/validate.py confirm
~/okc/bin/python experiments/gfuse_integration/validate.py fresh
~/okc/bin/python experiments/gfuse_integration/validate.py scored
~/okc/bin/python experiments/gfuse_integration/counters.py
python experiments/gfuse_integration/audit.py
```

Custom drivers lock generation, reference, capture and replay. Every timed
replay follows a 512 MB L2 flush. Each case has 80 adjacent AB/BA pairs; reported
source medians subtract the empty-graph median. Paired differences need not
match differences of separate medians. Confirmation changes the seed and
reverses capture order. Source files are not modified by the drivers.

## H100 results

Paired integrated-minus-previous times, microseconds:

| Full verify case | Initial | Second seed / reversed capture |
|---|---:|---:|
| B16 80K page1 | -1.360 | -0.976 |
| B32 80K page1 | -1.264 | -0.944 |
| B8 30K page1 | -1.728 | -1.760 |
| B12 80K page1 | -0.416 | -0.816 |
| B20 6K page64 scaled edges | +0.128 | +0.160 |
| B16 20K page64 scaled edges | -0.352 | -0.400 |
| B2 DP 12K page64 scaled edges | -0.048 | -0.272 |
| B4 DP 20K page1 soft/mixed | -1.600 | -1.024 |

The small B20 regression is within observed unchanged-control variation
(-0.224..+0.288 us over these two passes); no nearby cutoff was fitted.
B64 40K mixed/page1 improved by 13.296 us in one pass, not independently
confirmed. Results support integration, not uniform nearby superiority or
H200 qualification. The all-scored pass gives -0.320 / -1.168 us for verify16/32 full;
unchanged controls range -0.128..+0.112 us. Full data: pairs-scored.json.
The standard 12-shape bench gave 2.018x against fixed SGLang/H200 times;
this cross-device descriptive number is not a judge result.

## Validation scope

The standard suite passed 16/16. Contrary to the older tools/gluon_cases.py
introductory comment, its two DP verify cases do reach Gluon; the scored TP8
B16/B32 cases require additional coverage. Paired runs cover 15 initial,
12 confirmation and 12 scored shapes, with independent output/cache checks
before capture, after capture and after timing for both sources. Raw timings
and counts are audited, not just rounded summaries.

Fresh replay tests change QKV, FP8 caches, routing, lengths, positions and
new cache locations in already captured graphs. Each of 15 cases receives
four fresh inputs per source, including halved and empty history: 120 replays.
Cases include one-wave and two-wave plans, native controls, TP8/DP, page1/64,
scales1/1/64, and soft/coherent/edge distributions.

Counter stress uses three sequentially interleaved captured graphs at B12,
B16 and B32. Fresh inputs replace buffers before every replay. Initial upper
counter words 0, 0x7ffe and 0xfffe exercise ordinary reuse, signed-bit crossing
and 16-bit generation wrap. Every one of 36 calls checks output/cache against
the reference, verifies lower arrival bits are zero, and checks exact per-pair
generations. Counter seeding/readback belongs only to this offline driver.
No concurrent-stream safety, sanitizer, SASS occupancy or H200 claim is made.

No judge requests: Codex's three scored runs are already used today and the
last shared check is preserved. Exact integrated source needs its own H200
check before a future scored submission. No diagnostic code enters submission.py. Late main e6390f7 adds notes/tools
only and was also merged as 0acb168; the validated kernel hash is unchanged.
