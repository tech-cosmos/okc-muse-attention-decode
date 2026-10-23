Native Hopper attention experiment, Codex session 1, 2026-10-23.

This branch intentionally does not change submission.py. The native kernel loses
on all three measured full-attention verify shapes. Final graph replay medians:
  verify 1:  34.13 us native,  18.83 us Triton
  verify 16: 235.97 us native, 161.09 us Triton
  verify 32: 449.06 us native, 304.90 us Triton
The comparison uses the baseline inherited from origin/main at 8743290.

The retained module is v13 with explicit asynchronous-register compiler fences
and extended gather depths through seven stages (which did not improve it).
It is a research starting point, not a submission candidate.

Reproduce from the repository root:
  . /home/daytona/okc/bin/activate
  MAX_JOBS=2 python experiments/bench_cuda.py test
  MAX_JOBS=2 python experiments/bench_cuda.py bench
  MAX_JOBS=2 python experiments/bench_cuda.py sweep
  MAX_JOBS=2 python experiments/bench_cuda.py nearby

The module sets sm_90a explicitly and builds beneath experiments/.build.
GPU timing uses tools/bench.py's flock, cold L2, graphs, and empty-replay correction.
Use flock /tmp/okc-gpu.lock for profiler and sanitizer runs as well.

Validation: all 16 local test cases passed under native/fallback dispatch (the
five full-verify cases exercise native code); seven extra native cases passed,
including a 1/64 cache scale. Memcheck and synccheck report zero errors on six
small cases with pipeline depths 3, 5, and 7. Saved logs are in results/.

archive/ contains historical variants. They need TORCH_CUDA_ARCH_LIST=9.0
(v1-v7) or 9.0a (v8 onward) if replayed as standalone replacements. Some use the
old default torch extension cache rather than the retained module's local build.
In the v11 experiment, stage label 4 aliases 3 to stay below static shared-memory
limits. v2 timing overlapped Nsight and should be discarded. Later profiling
used the shared GPU lock. The v1-named profile is actually v2: the process
imported after the source edit; its export is named cuda-v2-profile.txt.

The handoff and next directions are in NOTES-codex.md on the codex branch.
