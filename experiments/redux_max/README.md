# Session 40: ordered integer warp maximum in native full merge

Rejected as a production optimization: exact maximum and less dependent shuffle
work, but no consistent complete-graph improvement. Production `submission.py`
was never edited. Frozen baseline is codex 023f00c, source d125df9, SHA256
ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca.

`generate.py` changes only the full-native merger's `max32` helper and extension
name. For finite FP32 bits b, the signed integer b XOR ((b >> 31) & 0x7fffffff)
preserves numerical order. Apply `__reduce_max_sync` and the same inverse bit
transform. This does not quantize floats or approximate their maximum. No NaN
input semantics are proposed; the merge already clamps each input through fmax
with its finite sentinel. All 32 lanes reach the full-mask reduction together.
No tile, warp count, partial format, launch, PDL, precision or dispatch changes.
Every other Python AST node, including native sliding, is unchanged; all 20
main-owned helpers match pinned main06b7755.

H100 cold-L2 complete graphs, 80 adjacent AB/BA pairs/case. Confirmation uses a
second seed and reversed module/capture order. Times in microseconds:

| Full case | initial previous/candidate | initial paired delta | confirmation delta |
|---|---:|---:|---:|
| B1 L1 80K page1 | 16.992/16.688 | -0.192 | +0.016 |
| B1 L8 80K page1 | 18.208/18.208 | 0.000 | -0.080 |
| B1 L1 80K page64 | 16.768/16.768 | -0.048 | -0.096 |
| B1 L8 80K page64 | 17.824/18.048 | +0.176 | -0.144 |
| B2 L8 9K page64 scaled edges | 10.032/9.904 | -0.160 | -0.080 |
| B2 L8 9K page1 scaled edges | 11.280/11.136 | -0.128 | +0.096 |

No positive cutoff or further sweep selected. Separate medians need not subtract
to paired deltas. The first decode improvement disappears; near-shape signs
vary. Standard 12-shape bench aggregate 2.005x versus fixed SGLang/H200 times is
unpaired local evidence, not H200 qualification or production regression.

Evidence:

- Full candidate suite 16/16, then all12bench. Only two suite cases reach the
  changed native full merge; fourteen are unchanged controls.
- 16 initial +6confirmation cases,1760adjacent pairs,132independent output/cache
  checks.128fresh wrapper replays (64/source),44changed-native plus44old-native;
  other calls are unchanged controls. Fresh QKV/cache/routes/lengths/positions/
  cache slots, shortened and empty history, pages1/64, scales1/1/64.
- Isolated native72fresh replays across S1/16/64/65/76/128, L1/8, TP8/DP,
  PDL on/off, empty groups, maxima positive/negative/zero and +/-1e30, numerator
  magnitudes0 through4.48e8, denominators through1e6, scales1/256,1/64,1,8.
  Float64 reference maximum relative L2 0.00177221 including BF16 output rounding.
- Synccheck separately repeats all72 isolated replays: zero errors. No new
  memcheck/racecheck/NCU claim; addresses and memory synchronization unchanged.
- `bitprobe.py`:1049984sampled finite FP32 values in32812rows, including negative
  values, subnormals, extrema, ties and signed zeros. Exact CPU bit roundtrip and
  numeric maximum; three GPU graph replays with rotated lanes give exact numeric
  maxima. No signed-zero bit identity or NaN-input claim.
- A synthetic DRIVER edit initially put denominator initialization in the rep2
  conditional, leaving it uninitialized in rep0. Retained failing source/log as
  `synthetic-driver-init-failure.*`. Fixed indentation before final tests. The
  separate16-call `debug_extreme.py` compares baseline/candidate with zero
  numerators at eight maxima; both produce finite zeros. No kernel failure was
  hidden and no invalid result used for timing or promotion.
- Both PDL variants retain48registers,3200reported shared bytes,zero local/stack.
  SASS has one REDUX.MAX.S32, five fewer SHFL and five fewer FMNMX sites.
  Counts including padding:536/544 ->536/536, non-PDL/PDL. All17coherent PDL
  loads remain after ACQBULK. No instruction-throughput or phase-duration claim.

Reproduce with ~/okc activated, from repository root:

    python experiments/redux_max/generate.py
    python experiments/redux_max/validate.py suite
    python experiments/redux_max/validate.py compare
    python experiments/redux_max/validate.py confirm
    python experiments/redux_max/validate.py fresh
    python experiments/redux_max/synthetic.py plain
    python experiments/redux_max/bitprobe.py
    /usr/local/cuda-13.0/bin/compute-sanitizer --tool synccheck --error-exitcode 17 python experiments/redux_max/synthetic.py synccheck
    python experiments/redux_max/disassemble.py
    python experiments/redux_max/audit.py

Custom GPU drivers hold the shared lock over generation/reference/capture/replay,
flush512MB before measured graphs and subtract empty replay cost. Suite holds
lock; standard benchmark uses its existing timing-loop lock. All own GPU work
sequential. Disassembly is CPU-only and may overlap validation. No judge jobs.
