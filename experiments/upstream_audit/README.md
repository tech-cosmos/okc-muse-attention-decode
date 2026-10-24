# Session 41: native contribution after upstream BF16 partials

Decision: retain production unchanged. This is a revalidation of the existing
native dispatches against current upstream, not a new kernel or a new speedup.
No judge check or scored request. Source snapshot `native.py` is exact production
SHA ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca;
`upstream.py` is origin/main 463b0d5, whose kernel is 06b7755. Its SHA is in audit.json.
The comparison includes main's BF16 partial policy versus native's necessary FP32
full-merge interface. It measures the whole dispatch choice, not merger latency alone.

Run from the repository root in ~/okc:

```sh
python experiments/upstream_audit/validate.py compare
python experiments/upstream_audit/validate.py confirm
python experiments/upstream_audit/validate.py fresh
python experiments/upstream_audit/validate.py scored
python experiments/upstream_audit/audit.py
```

In raw records, `previous` means upstream and `merged` means native production.
A negative paired delta favors native. Neither source is modified by the driver.
`confirm` changes the seed and reverses module/capture order. All modes run
sequentially; each owns /tmp/okc-gpu.lock across generation, reference, capture
and replay. Every measured graph gets a 512 MB flush; an empty graph's median
is subtracted from each source's reported median. Raw pair deltas do not require
that subtraction and need not equal differences of separate medians.

## Results (H100, microseconds)

| Shape | Initial paired delta | Confirmation | Scored-shape pass |
|---|---:|---:|---:|
| B1 L1 sliding, 80K/page1 | -0.352 | -0.384 | -0.432 |
| B1 L1 full, 80K/page1 | -0.608 | -0.656 | -0.704 |
| B1 L8 full, 80K/page1 | -0.352 | -0.128 | -0.288 |
| B1 L1 full, 80K/page64 | -0.128 | -0.624 | — |
| B1 L8 full, 80K/page64 | -0.656 | -0.848 | — |
| B2 L8 full, 9K/page64, scaled edges | -1.216 | -1.440 | — |
| B2 L8 full, 16K/page64, soft | -0.496 | -0.208 | — |
| B8 L1 sliding, 80K/page1 | -0.336 | -0.416 | — |
| B8 L1 sliding, 80K/page64 | -0.576 | -0.768 | — |

All 24 initial native cases and 12 repeated native cases favor native, with
varying effect sizes. Three unchanged controls have initial paired offsets
-0.032 to 0 us and confirmation offsets -0.192 to +0.096 us. The nine unchanged
scored controls range -0.320 to +0.048 us. Small effects, particularly B1 full
verify's confirmation, are comparable to control variation. These are local
measurements, not evidence of a new H200 rank or uniform superiority nearby.
The old page64 weakness from sessions 26/27 used an older upstream and different
inputs; this audit does not isolate why it differs. No page/context cutoff is fitted.

## Coverage and limits

27 initial cases, 15 confirmation cases and all 12 scored shapes, each 80
adjacent AB/BA pairs: 4,320 pairs and 324 independent output/cache checks.
Native initial cases: 14 full merge and 10 sliding; three unchanged controls.
Fresh replay coverage: all 27 cases, four changed inputs per source, 216 total.
For native production, 56 full-merge calls, 40 sliding calls and 12 fallback
calls; upstream supplies 108 separate comparison calls. QKV, FP8 cache, routes,
lengths, positions and cache locations change. Halved and empty histories are
included. Cases cover pages 1/64, scales 1/1/64, TP8/DP, soft/coherent/edges,
contexts 10–120K and small-batch dispatch boundaries. All checks pass.

Production did not change: no redundant 16-test suite, sanitizer or disassembly
run, and no new claims of such coverage. Session 36 exact-source validation
still applies. The CPU audit verifies snapshots, all shared function/class ASTs
except the intentionally different dispatcher, raw pair counts and medians,
and reference/fresh check counts. Existing native kernels, FP32 guard and all
upstream helper definitions remain unchanged. No experiment is imported by
submission.py. Budget snapshots and leaderboard state are archived.

Future work should use these snapshots as the current upstream comparison.
Do not retest the same unchanged source next cycle without new evidence or a
main kernel update. The current code still needs its own qualifying H200 check
before any future scored submission; today's Codex scored allowance is exhausted.
