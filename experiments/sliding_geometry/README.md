# Session 44: native sliding address arithmetic

Rejected both candidates. Production stays at SHA256
`a7a3db9737aa25b42695a31b4c16d4da3649d910b2366bcfc18453a6a5a9b9e3`.
Parent `d41438a` includes main `e1b9a40` (notes only since session 43).
No judge requests, new dispatch, precision change or Triton tuning.

`candidate.py` specializes the embedded native sliding kernel on HKV=2/HQ=16
and HKV=16/HQ=128, retaining runtime geometry otherwise. This makes the cache
row stride and fused projection width compile-time constants. `unsigned.py`
additionally uses unsigned thread/lane and gather-chunk indices, removing
signed division/remainder correction sequences for nonnegative topology indices.
All cache addressing still uses the original 64-bit row address expression.
Every key, cache write, matrix operation, barrier and launch remains present.

For page1/early-gather, static instruction sites are 1056 (production), 1040
(geometry), 928 (geometry + unsigned indices). Sites before the first LDGSTS
are 109/100/79. All use 74 registers, 17408 reported shared bytes, zero stack
and local memory. Four generic geometry-only instantiations are instruction-text
identical to production. Each instantiation retains 4 LDGSTS, 3 BAR.SYNC,
2 QGMMA and 4 HGMMA sites. These counts do not measure critical-path latency.

H100 paired candidate-minus-production microseconds, initial / confirmation:

| Sliding L1 case | Constant geometry | Plus unsigned indices |
|---|---:|---:|
| B1 TP8 80K page1 (scored) | 0.000 / +0.032 | -0.016 / -0.032 |
| B2 TP8 80K page1 mixed | 0.000 / 0.000 | -0.064 / -0.048 |
| B8 TP8 80K page1 mixed | +0.192 / +0.208 | -0.064 / -0.224 |
| B1 TP8 80K page64 | +0.064 / +0.112 | +0.112 / -0.160 |
| B8 TP8 80K page64 mixed | +0.064 / -0.016 | -0.096 / -0.128 |
| B1 DP 6K page64 scaled edges | -0.032 / -0.032 | -0.224 / 0.000 |

The small scored unsigned difference is below unchanged-control variation.
Unsigned nearby results are mixed and do not justify a dispatch cutoff. Keep
both changes separate from production; do not reopen this address-specialization
family merely because it removes instructions.

Each candidate passes 16/16 standard cases and a subsequent 12-shape bench,
14 initial + 8 confirmation cases with 80 adjacent AB/BA pairs each, and
112 fresh graph replays (56 candidate, 56 production). Total: 3520 adjacent
pairs, 264 independent output/cache reference checks around timing, and 224
fresh replays. Fresh inputs change QKV, FP8 cache, routing, lengths, positions
and new cache slots; they include halved and empty histories. Twelve initial
cases exercise native code and two are unchanged controls. Pages 1/64,
TP8/DP, scales 1 and 1/64, and soft/edge cases are covered. Standard suites
exercise three changed native cases each; explicit cases cover DP and scored
TP8. No new sanitizer, profiler, runtime-generic-geometry or H200 claim.

Comparisons hold the shared GPU lock across generation, references, capture,
and replay. Each timed graph follows a 512MB L2 flush; empty replay is subtracted.
Confirmation uses a second seed and reverses capture order. Separate medians
need not subtract to the paired median delta. The standard bench has its own
timing-loop lock; suites hold the lock. Own GPU work ran sequentially.
Local benchmark aggregates (2.026x geometry, 1.998x unsigned) use fixed SGLang
H200 numbers and vary on unchanged paths; they are not candidate speedups or
judge qualification.

Reproduction from the repository root, after activating `~/okc`:

```sh
python experiments/sliding_geometry/generate.py
python experiments/sliding_geometry/validate.py suite
python experiments/sliding_geometry/validate.py compare
python experiments/sliding_geometry/validate.py confirm
python experiments/sliding_geometry/validate.py fresh
VARIANT=unsigned python experiments/sliding_geometry/validate.py suite
VARIANT=unsigned python experiments/sliding_geometry/validate.py compare
VARIANT=unsigned python experiments/sliding_geometry/validate.py confirm
VARIANT=unsigned python experiments/sliding_geometry/validate.py fresh
python experiments/sliding_geometry/audit.py
```

`audit.py` verifies exact generation, all ASTs outside the native sliding
extension, production identity, assembly counts and generic equivalence,
copy-topology bijection, raw medians and validation counts. Sources, CUDA,
disassembly/resource reports and raw results are archived here. The initial
comparison used the same driver before its output-directory/variant selector
was added; its JSON was moved into `results/candidate` after it finished.
Its seeds, shapes, captures and timing logic are unchanged.
