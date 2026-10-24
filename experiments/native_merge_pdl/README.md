# Session 27: small-full native merge launch diagnosis

No production change, dispatch cutoff, or judge request. Snapshots are exact
production d8bab0a (SHA 1bcf42aa6edf) and main 824e52a (SHA 4f9ccd6bab27).
This follows the session 25/26 narrow B2/L8/page64 native-merge weakness.

## Experiment

`compare.py` uses four complete kernel functions: main/native with their existing
PDL merge launches, and serialized-launch controls for each. The native control
selects the existing `combine_full(..., pdl=False)` specialization. The main control
changes only its merge's `PDL` and `launch_pdl` flags in an offline wrapper. All
attention, plans, precision, keys, cache updates and output work are preserved.
There are no production imports or Triton/Gluon source/plan edits.

20 distinct shapes: both pages, B1 decode/verify 80K, B2 verify agentic
8192/9K/10K/16K/40K/80K, B2 fixed9K, B1 fixed9K. Every case actually dispatches
native merge (65–128 slots). Four adjacent AB/BA contrasts per shape,80 pairs each;
eight second-seed confirmations at120 pairs each. 10240 adjacent pairs total.
Each contrast has its own paired control. Do not subtract independent medians.
All custom drivers hold /tmp/okc-gpu.lock through generation/reference/capture/
replay. Each timed graph follows a512MB flush; an empty replay is subtracted.
Bootstrap intervals resample AB/BA blocks and describe only within-run variation.

Paired deltas, H100 microseconds, initial / confirmation:

| case | native minus main, PDL | serial native minus native PDL |
|---|---:|---:|
|B1 L1 80K page1 (scored)|-0.048/-0.064|+0.800/+0.608|
|B1 L8 80K page1 (scored)|-0.464/-0.496|+0.656/+0.592|
|B2 L8 8192 agentic page1|-0.912/-1.152|+0.416/+0.736|
|B2 L8 9K agentic page1|-1.184/-1.088|+0.800/+0.672|
|B2 L8 16K agentic page1|-0.880/-0.320|+0.192/+0.288|
|B2 L8 8192 agentic page64|+0.096/-0.128|+0.512/+0.480|
|B2 L8 9K agentic page64|+0.192/+0.128|+0.032/+0.144|
|B2 L8 16K agentic page64|+0.416/+0.208|+0.224/+0.480|

Native serialization fails as a remedy and loses on both scored small-full
shapes. At9K/page64, serial main versus serial native differs by-0.096/-0.160us,
while page1 is+0.032/-0.064us: much of the difference depends on the concurrent
launch chain. This does not by itself isolate register contention, load placement,
wait cost, scheduling policy or code-generation effects. PDL flags change both
launch behavior and the compiled wait specialization. No threshold is fitted.

## Timeline and residency evidence

`timeline.py` uses Nsight Systems node tracing for B2/L8/9K at pages1/64,
all four modes,three repeats:24 complete graph replays. These instrumented
latencies are not benchmark times. `timeline-intervals.json` preserves raw
NVTX bounds and kernel timestamps/resources; `timeline-summary.json` has medians.

At S76 and304 attention CTAs, page1 main/native merge starts6.368/6.272us
after attention starts; page64 starts0.864/0.960us after. PDL overlap is
3.840/3.648us on page1 and7.520/8.032us on page64. Serialized controls show
zero overlap; both serialized merges take about3.2us in this trace. Profile
rows include waits and must not be summed as graph latency.

`occupancy.py` queries cuOccupancyMaxActiveBlocksPerMultiprocessor for the
compiled, unchanged attention functions. Page1 uses170 registers,62464 shared
bytes, **2 maximum active CTAs/SM**; page64 uses168 registers,same shared bytes,
**3 CTAs/SM**. On132SMs,304 CTAs exceed page1's264 slots but fit page64's396.
This is theoretical occupancy, not achieved occupancy. The native merge has
256threads/48registers, the main merge128threads/56registers; both have256CTAs.
All four modes have identical attention SASS for each page, verified by hashes,
despite different Triton compilation-cache hashes.

The capacity difference and trace support an overlap/scheduling explanation for
the page-dependent weakness. They do not establish a universal cutoff or prove
an isolated hardware cause. No new architecture is justified by these data alone.

## Validation

- 336 independent output/cache checks in timing drivers (4 modes,3 checkpoints,
 20 initial+8 confirmation entries).
- 320 fresh graph replays (20 cases,4 new inputs,4 modes), including changed
 QKV/cache/routes/lengths/positions/cache slots, shortened and empty contexts.
- 16 independent output/cache checks for the timeline and8 for the final
 occupancy capture. Initial occupancy collection was rerun while adding SASS
 evidence; counts above refer to final retained drivers only.
- The first SASS hash parser expected cuobjdump addresses, but Triton's disassembler
 emits decoded lines. It hashed empty strings. Those hash fields in
 occupancy-invalid-hash-parser.* are invalid; occupancy/output observations were
 unaffected. The final driver normalizes trailing whitespace and hashes the full saved SASS, and the CPU audit checks
 actual file bytes and equality. No timing result depends on the invalid hashes.
- No native implementation edit, new sanitizer claim, or redundant unchanged
 production16-case suite. Session26 production validation still applies.

## Reproduce

From repository root, activate ~/okc. CUDA13 tools are under /usr/local/cuda-13.0/bin.

```sh
python experiments/native_merge_pdl/compare.py compare --rounds 80 --name initial
python experiments/native_merge_pdl/compare.py compare --rounds 120 --seed 27203 --name confirmation --indices 0 1 2 3 5 12 13 15
python experiments/native_merge_pdl/compare.py fresh --name fresh
/usr/local/cuda-13.0/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o experiments/native_merge_pdl/results/timeline python experiments/native_merge_pdl/timeline.py
/usr/local/cuda-13.0/bin/nsys export --type=sqlite --force-overwrite=true --output=experiments/native_merge_pdl/results/timeline.sqlite experiments/native_merge_pdl/results/timeline.nsys-rep
python experiments/native_merge_pdl/summarize_timeline.py
python experiments/native_merge_pdl/occupancy.py
python experiments/native_merge_pdl/audit.py
```

Redirect stdout/stderr to the corresponding results/*.log paths when reproducing
the audit. Binary nsys/sqlite artifacts stay ignored; readable raw intervals,
SASS, logs, pairs, sources and audit are committed. No checks or scored runs used.
