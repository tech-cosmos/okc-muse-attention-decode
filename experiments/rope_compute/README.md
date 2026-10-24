# Session 34: computed RoPE in native sliding attention

This experiment replaces a position-dependent global table lookup with explicitly
range-reduced hardware sine/cosine. It changes only the native sliding L1 path,
keeping the existing shape predicate, attention, split plan, PDL merge, FP8
query/cache precision, FP16 PV inputs, FP32 partials, and BF16 output. It is separate
from production and is not submitted to the judge.

The motivation is session32's large metadata/preparation interval. Unlike Claude's
earlier libdevice trigonometry experiment, both implementations explicitly reduce
angles before executing PTX `sin.approx.ftz.f32` and `cos.approx.ftz.f32`. This is
also distinct from the closed query-layout and load-placement experiments.

## Arithmetic and configuration

The frequency vector has 32 FP32 values, constructed with the same Torch operations
as the original table and cached only by configuration and device. No input-derived
data is cached. The kernel loads frequencies through position-independent
addresses and computes the same FP32 `position * frequency` as the table.

- `candidate`: round the number of turns and reduce with an FP64 FMA, then round
  the reduced angle to FP32.
- `compensated`: compute the turn count in FP32 and subtract two FP32 pieces of
  2*pi with two explicit round-to-nearest FMAs. This avoids FP64 arithmetic and
  conversions while retaining cancellation accuracy.

The hardware functions receive a reduced angle near [-pi,pi]. NVIDIA documents
absolute error bounds for these PTX functions within [-2*pi,2*pi]; see the
[PTX sine and cosine specifications](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin).
No large-argument fast-intrinsic accuracy assumption is used. The integer turn
conversion is intended for the original table's position domain, 0 through
2^21-1. Neither variant expands the production-supported position domain.

`accuracy.py` exhaustively checks all 2^21 positions x 32 frequencies for theta
500000, 10000 and 1000000 against Torch's FP32 sine/cosine of the same FP32 angle.
The independent diagnostic calls the exact helper compiled into each candidate.
Complete attention checks also cover alternate theta configurations and contexts
up to 2^21-1. The accuracy test does not substitute for output/cache checks and
does not claim bit-identical FP8 rounding at every boundary.

## Reproduction

Use the existing `~/okc` venv and run from the repository root. Set
`TORCH_EXTENSIONS_DIR=$PWD/experiments/rope_compute/build` and `MAX_JOBS=4`.
`ROPE_VARIANT` is `candidate` by default, or `compensated`.

```sh
python experiments/rope_compute/generate.py
python experiments/rope_compute/accuracy.py
python experiments/rope_compute/run.py suite
python experiments/rope_compute/run.py compare --rounds 60 --name initial
python experiments/rope_compute/run.py fresh --name fresh
python experiments/rope_compute/config_check.py
python experiments/rope_compute/validate_remaining.py
python experiments/rope_compute/collect_profiles.py
python experiments/rope_compute/audit.py
```

Run GPU drivers sequentially. `validate_remaining.py` records the candidate's
alternate-config/sanitizer runs, then the compensated suite/bench, comparisons,
fresh checks, alternate configurations and sanitizers. Run the compensated
arithmetic diagnostic separately with `ROPE_VARIANT=compensated`.

Custom drivers hold `/tmp/okc-gpu.lock` across generation, reference checks,
capture and replay. Each timed graph has a 512MB cache flush, with empty-replay
subtraction. Complete parent/candidate graphs use adjacent AB/BA pairs; a second
capture of identical production provides an allocation/capture control. Every
compared graph is reference-checked before capture, after replay and after timing.
The full benchmark retains bench.py's timing-loop locking; it runs sequentially
with this experiment's other GPU work. Profiling acquires the lock in the parent
before starting NCU, after all correctness and timing processes finish.

Sanitizers launch native attention only, and Torch independently reduces FP32
partials and checks the final output and cache. They exercise fresh values/routes
and empty contexts, pages 1/64, scales 1 and 1/64, and contexts 10/257/4096.
Runtime tests use EARLY=True; EARLY=False is compiled only. No synchronization,
gather coverage or matrix-product precision was modified.

`production_snapshot.py` is frozen SHA256
`1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548`,
including main 824e52a. The generator makes all native changes explicitly. The CPU
audit regenerates sources, checks unchanged production/shared helpers/dispatch,
recomputes paired deltas, validates result counts and checks SASS/resources.
Disassembly whitespace is normalized; compiled sources receive no post-test edits.

## Results

See `results/audit.json` for complete timing records, validation counts, resource
checks and arithmetic error maxima. The session handoff in `NOTES-codex.md` records
the decision and the selected paired measurements. Raw graph pairs are retained
in `initial.json` and `compensated-initial.json`. Full bench aggregates compare
H100 measurements with fixed SGLang/H200 values and are not H200 speedup estimates.

Both candidates lose on all 18 measured shapes; neither is promoted.

| Case | FP64 paired delta (us) | Compensated FP32 paired delta (us) |
|---|---:|---:|
| B1 / 80K / page1 | +0.512 | +0.448 |
| B8 / 80K / page1 | +0.864 | +0.608 |
| B1 / 80K / page64 | +0.640 | +0.512 |
| DP1 / 4096 / page64 / scale1/64 | +0.704 | +0.512 |
| B1 / 2^21-1 / page1 | +0.576 | +0.512 |

Each variant passes 16/16 (3 native, 13 fallback), all 12 benchmark shapes,
72 fresh native graph replays and 12 alternate-configuration output/cache checks.
Both independently pass native memcheck/synccheck with zero errors and racecheck
with zero hazards, errors or warnings (24 fresh replays per tool and variant).
There are 4,320 adjacent timing pairs and 324 reference checks in the paired
comparisons, plus three independent profiled output/cache checks.

Page1/EARLY static instruction sites are 1056/1136/1096 for production/FP64/FP32.
Each computed version replaces 10 global load sites with 10 sine and 10 cosine
sites plus range reduction; the FP32 control eliminates all FP64 instructions.
All use 74 registers and 17408 shared bytes, with no stack/local allocation.
The three CTA barriers and 2 QGMMA / 4 HGMMA sites remain.

NCU executed warp instructions: 164482 / 172674 / 168578.
This supports added arithmetic work, without isolating its critical-path
latency or attributing the entire graph loss to a specific instruction.
Profiler durations are not graph timings. Identical-production capture offsets
range from -0.144 to +0.192 us. No favorable-repeat selection, dispatch cutoff,
further arithmetic sweep, H200 request or H200 speedup claim is justified.
