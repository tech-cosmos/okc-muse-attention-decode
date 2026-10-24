# Session 38: denominator folded into native RMSNorm

Rejected local experiment. Production stays ce81a25 (kernel d125df9), SHA256
ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca.
Main06b7755 was already integrated; no shared Triton helpers or plans changed.
No judge requests. The exact production source still lacks an H200 check.

For merged numerator n, positive denominator d, cache scale v and epsilon e,
the existing merge computes a=n*v/d, then a/sqrt(mean(a*a)+e).
The candidate uses n*v/sqrt(v*v*mean(n*n)+e*d*d). This is identical in real
arithmetic, retains the entire denominator and epsilon, and keeps every
activation/partial in its existing precision. It removes denominator dependence
from the vector square/reduction path and one reciprocal from generated SASS.
There is no BF16 partial-storage change, attention change or dispatch change.

Cold-L2 H100 complete CUDA graphs, 80 adjacent alternating AB/BA pairs/case;
negative paired delta means candidate faster. Confirmation changes seed and
reverses module/capture order. Labels `merged` in raw files mean candidate.

| Full case | initial old/candidate us | paired delta us | confirmation delta us |
|---|---:|---:|---:|
| B1,L1,80K,page1 | 17.040/17.024 | -0.096 | +0.176 |
| B1,L8,80K,page1 | 18.528/18.384 | -0.096 | -0.096 |
| B1,L1,80K,page64 | 16.672/16.816 | +0.224 | -0.016 |
| B1,L8,80K,page64 | 17.856/17.952 | +0.064 | -0.160 |
| B2,L8,9K,page64,scaled edges | 10.064/10.016 | -0.032 | -0.032 |
| B2,L8,9K,page1,scaled edges | 11.360/11.264 | -0.096 | +0.128 |

The small mixed deltas do not establish a complete-graph gain; no favorable
shape cutoff or precision variant was promoted. Separate medians need not
subtract to the median of paired differences. Unchanged controls have offsets
up to 0.112us in the initial run. No H200 timing inference is made.

Validation:

- Full reference suite16/16, then all12bench shapes; local aggregate2.001x versus
  fixed SGLang/H200 times, not an H200 estimate. Only2 suite cases exercise the
  changed native merger; the other14 are controls.
- 16initial+6confirmation cases,1760adjacent pairs,132independent output/cache
  references. Eleven initial cases and all6confirmation cases reach the native
  merger; the remaining5initial cases retain unchanged paths.
- 128fresh full-wrapper graph replays,64/source, including44changed-native
  calls plus44old-native controls. Fresh QKV/cache/routes/lengths/positions/
  cache slots, shortened/empty history, pages1/64, scales1/1/64 and different
  distributions. Captured dispatch remains based only on metadata.
- Isolated native merge:12cases x6fresh graph replays =72. PDL on/off,
  S1/16/64/65/76/128,L1/8,TP8/DP,empty split groups, vector scales0/1e-8/1e-3/
  1/1e5/4.48e8, denominator values through1e6, cache scales1/256,1/64,1,8.
  Float64 independent reference; maximum relative L2 0.00176830 including BF16
  output rounding. Zero numerators produce exactly zero outputs.
- Initial synthetic driver stopped on assert_close's dtype check (actual FP32
  versus reference FP64), not a numerical assertion. Failed log is retained;
  check_dtype=False permits intentional comparison to the higher precision
  reference. All72 replays then pass. No failed timing was used.
- Both PDL/non-PDL specializations remain48registers,3200reported shared bytes,
  no local/stack. Static reciprocal sites7->6, EX2 sites11 unchanged.
  Static instruction sites including padding: non-PDL536->536,PDL544->536.
  All17coherent partial-load sites follow ACQBULK in the PDL version.
- CPU audit reproduces candidate from frozen parent, asserts every other AST
  node identical, validates result counts/raw medians and SASS ordering/resources.
  No new sanitizer or NCU claim; this is a final arithmetic-only change.

Custom drivers hold /tmp/okc-gpu.lock through generation/reference/capture/
replay, flush512MB before each timed graph, and subtract an empty replay.
Suite holds the lock; standard bench uses its timing-loop lock. Own GPU jobs
run sequentially. No work derived from input escapes the timed kernel.

Reproduce from repo root with ~/okc/bin/python, CUDA_HOME=/usr/local/cuda-13.0
and TORCH_EXTENSIONS_DIR=$PWD/.build/s38:

```sh
python experiments/homogeneous_norm/generate.py
python experiments/homogeneous_norm/validate.py suite
python experiments/homogeneous_norm/validate.py compare
python experiments/homogeneous_norm/validate.py confirm
python experiments/homogeneous_norm/validate.py fresh
python experiments/homogeneous_norm/synthetic.py
python experiments/homogeneous_norm/audit.py
```

The audit expects the checked-in frozen logs/results. Redirect new runs to
those filenames if regenerating evidence. SASS/resource dumps come from
cuobjdump --dump-sass/--dump-resource-usage on the corresponding native .so.
