# Session 36 late main integration

Pinned main06b7755 and merged as d125df9 after the BF16 integration finished.
The inherited Gluon producer/consumer removes duplicate barriers before
partition mbarrier arrivals; each consumer retains local FP32 row sums and
reduces them at its epilogue. Codex did not tune these kernels or plans.
All20 shared helpers match main06b7755 exactly. Native extensions and kernel_fn,
including the FP32 native-full partial guard, are unchanged from23432d0.

Final source SHA256:
`ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca`.
The exact previous/current/main snapshots and reproducible audit are here.
This source has no exact-source H200 qualification. No judge requests.

Validation:16/16, then all12benchmark shapes; local aggregate2.014x against
fixed SGLang/H200 times. It is unpaired, not an H200 estimate. Twelve
previous/current cases have960adjacent AB/BA pairs and72independent output/cache
checks, plus96fresh graph replays (48per source). Thirty-six of the48current
fresh replays exercise changed Gluon paths; twelve cover unchanged controls.
Cases include batches5/6/8/12/16/32/64, TP8/DP, pages1/64, scales1/1/64,
soft/coherent/edges and shortened/empty contexts. QKV/cache/routes/lengths/
positions/cache slots change. No new native or integrated sanitizer claim.

| Full verify case | Previous us | Integrated us | Paired delta us |
|---|---:|---:|---:|
| B6,80K,page1 |69.376|69.024|-0.272|
| B16,80K,page1 |133.712|133.552|-0.160|
| B32,80K,page1 |255.920|255.568|-0.400|
| B64,40K,page1 |256.992|256.144|-1.200|
| B6,4096,page64,scaled edges |17.344|17.344|+0.016|

Most differences are small; no universal improvement or statistical H200
claim. B5full/native B1full/native B1sliding unchanged-control deltas are
-0.096/-0.048/-0.032us. Separate-median differences need not equal paired
medians. Do not add these deltas to the preceding integration's measurements.

Custom drivers hold the shared GPU lock over generation/reference/capture/
replay, flush512MB before every timed graph and subtract empty replay.
Standard benchmark retains its timing-loop lock. All own GPU work runs
sequentially. `audit.py` checks frozen sources, native identity, helpers,
complete results and recomputed medians.

```sh
python profiles/session36-late-integration/validate.py suite
python profiles/session36-late-integration/validate.py compare
python profiles/session36-late-integration/validate.py fresh
python profiles/session36-late-integration/audit.py
```

The native denominator experiment in e2c54fd is separate. Claude independently
implemented the same local-sum principle in its Gluon lane; this integration
inherits that implementation, not the rejected native prototype.
