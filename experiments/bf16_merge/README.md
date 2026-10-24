# Session 37: native BF16 partial merge

Rejected as a production change. The native small-full merger can correctly
consume BF16 partial storage, but removing production's FP32 compatibility guard
regresses scored decode and does not give a reliable verify improvement across
nearby cases. Production remains `d125df9`, SHA256
`ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca`.
No judge requests. Main `06b7755` was already integrated at both fetches.

## Mechanism

Main's inherited `pdt` policy selects BF16 split accumulators on its Triton full
paths. Codex production overrides that to FP32 when using its existing native
small-full merger, whose interface requires `data_ptr<float>()`.

This experiment templates the native merger over FP32/BF16 storage and removes
that guard. BF16 pairs use a coherent 32-bit global load after the PDL dependency
wait, widen exactly, and accumulate in FP32. Max/sum storage, attention arithmetic,
cache/query formats, new-token handling, launch/grid, eight-warp reduction and
shape coverage remain unchanged. All 20 Python helpers match pinned main;
no Triton plan or kernel is tuned. The dtype check reads tensor metadata only.

Three implementations:

- `initial.py`: widen loaded pairs with CUDA BF16 conversion intrinsics.
- `bits.py`: replace an observed redundant shift with direct exact bit widening.
- `candidate.py`: retain each pair as one packed register until its weighted
  accumulation. `generate_packed.py` reproduces this variant from `bits.py`.

BF16 storage is the same interpretation already adopted on main; this is not an
organizer ruling about scratch precision. No experimental dtype change is
promoted. Numerical validation alone is not treated as policy qualification.

## Results

H100 complete CUDA graphs, cold L2, 80 adjacent AB/BA pairs per case. Values below
are medians after subtracting an empty graph; the deltas are medians of paired
differences and need not equal differences of separate medians.

| Fixed 80K full case | Previous / direct bits, us | Paired delta | Previous / packed, us | Paired delta |
|---|---:|---:|---:|---:|
| B1,L1,page1 |16.944 /17.600|+0.704|16.976 /17.520|+0.528|
| B1,L8,page1 |18.640 /18.256|-0.432|18.496 /18.688|+0.224|
| B1,L1,page64 |16.640 /17.152|+0.384|16.544 /17.040|+0.464|
| B1,L8,page64 |18.000 /17.568|-0.480|17.952 /18.144|+0.224|

A second-seed packed
confirmation reverses module/capture order: paired deltas for those four cases
are +0.448/+0.336/+0.096/+0.256 us. B2,L8,20K,page64,scaled edges regresses
+0.736 us with direct bits and +1.984/+2.048 us with packed initial/confirmation.

Mixed-length verify cases do improve: packed B1,nominal80K,page1 saves1.696us,
and direct bits2.064us. Mixed lengths can be up to4x nominal context; these are
nearby cases, not scored80K measurements. The first variant's19-case comparison
contains mixed-length B1 shapes, not the fixed scored cases. The later23-case
runs explicitly add four fixed cases. No favorable cutoff is selected.

Each variant passes16/16 and all12 benchmark shapes. Local aggregates against
fixed SGLang/H200 times are2.014/1.986/2.014x; these unpaired totals are neither
H200 estimates nor evidence of improvement. Whole-graph paired results decide
rejection. The final candidate source SHA256 is
`bd81ce59842547fa75a1b72eaead0f29e8d6e12cb38dc8fab06ed9ac69898e79`.

## Evidence and scope

- 5,680 adjacent graph pairs and426 independent output/cache checks:
  19initial +23bits +23packed +6confirmation cases,80pairs each.
- 184fresh full-wrapper replays,92per old/final source across23cases. Of these,
  68final calls use the changed native BF16 merger, plus68old FP32 controls.
  Remaining cases exercise unchanged paths. Fresh QKV/cache/routing/lengths/
  positions/cache slots, including shortened and empty historical contexts;
  TP8/DP,pages1/64,scales1/1/64,soft/coherent/edges. See recorded native calls.
- The standard suite reaches the changed native full merger on2cases (splits112
  and127); its14other cases are controls, including the existing native sliding
  path. All benchmark cases are retained; no claim that16tests are all native.
- Isolated final native tests:24cases x3fresh graph replays =72, FP32/BF16,
  PDL on/off, splits1/16/64/65/76/128, L1/8, TP8/DP, empty partial groups,
  large unnormalized numerators and scales1/1/64. Independent Torch reduction
  checks finite output and rel-L2<0.003; maximum0.0017934 including BF16 output.
- Final candidate independently passes memcheck0errors, synccheck0errors,
  racecheck0hazards/errors/warnings, each72fresh isolated-native graph replays.
  No Triton loader kernels execute under these tools. These synthetic tests
  validate the merger, not an independent end-to-end attention implementation.
  No separate fresh/sanitizer claim for rejected initial/direct-bits versions.
- FP32 SASS is instruction-identical to previous production, both PDL modes.
  BF16 registers: initial/direct bits56(non-PDL)/57(PDL), packed32/32.
  All four specializations report3200shared bytes and zero local/stack.
  Packed BF16 has568static instructions versus624initial-PDL; coherent loads
  remain17sites, all after ACQBULK in both PDL variants. No occupancy or
  profiler-latency conclusion follows merely from the register reduction.
- Every custom driver locks generation/reference/capture/replay using the shared
  GPU lock, flushes512MB before each timed graph and subtracts empty replay.
  Standard bench uses its timing-loop lock; suite holds the lock. All own GPU
  work is sequential. No NCU profiling was needed. Only generated SASS trailing
  whitespace was normalized. CPU audit checks source identity, helper equality,
  generated-source equality, bit widening, SASS ordering, raw medians and counts.

## Reproduction

From repo root, activate `~/okc`, set `CUDA_HOME=/usr/local/cuda-13.0`, append its
`bin` to PATH, set `TORCH_EXTENSIONS_DIR=$PWD/.build`, and `MAX_JOBS=4`.

```sh
python experiments/bf16_merge/validate.py suite
python experiments/bf16_merge/validate.py compare
S37_REVERSE=1 S37_CONFIRM=1 python experiments/bf16_merge/validate.py compare 37572
python experiments/bf16_merge/validate.py fresh
python experiments/bf16_merge/synthetic.py plain
compute-sanitizer --tool memcheck --error-exitcode 99 python experiments/bf16_merge/synthetic.py memcheck
compute-sanitizer --tool synccheck --error-exitcode 99 python experiments/bf16_merge/synthetic.py synccheck
compute-sanitizer --tool racecheck --error-exitcode 99 python experiments/bf16_merge/synthetic.py racecheck
python experiments/bf16_merge/generate_packed.py
python experiments/bf16_merge/audit.py
```

For earlier variants set `S37_CANDIDATE=initial.py` or `bits.py`. To reproduce
initial comparison scope also set `S37_INITIAL_CASES=1`; set `S37_TAG=initial`
or `bits` for its raw output filename. Preserve archived logs when rerunning.

No production promotion; retain the FP32 native-interface guard. This closes
this uniform BF16-native-merge change, including early/direct/late widening.
Future work needs a new scheduling/data-movement mechanism, not another small
merge parameter or dtype-cutoff sweep. Current production still needs its own
exact-source H200 check before any future scored request.
