# Codex session 2: native sliding attention and output reduction

The production candidate is `submission.py` at commit **881df62**. Its only native
path is a small CUDA split reduction: one warp per query head, each lane computes
one of 32 split weights, and shuffles distribute weights for the FP32 accumulation.
It dispatches for sliding layers with 32 splits and B * Hkv <= 4. All other shapes
use the latest main (0495b16), including its batched Triton reduction. No Triton
plans were tuned by Codex.

Reproduce from the repository root, with `. /home/daytona/okc/bin/activate`:

```
python tools/bench.py test
python tools/bench.py bench
python experiments/sliding/final_pairs.py
python experiments/sliding/fresh_routes.py
python experiments/sliding/sanitize_final.py memcheck
python experiments/sliding/sanitize_final.py synccheck
```

All graph timing uses `tools/bench.py`: shared GPU flock, cold L2, graph replay,
median, and an empty-replay subtraction. `final_pairs.py` compares against the
archived main 0495b16 in the same process, alternating measurement order. Results
are in `results/`. Judge job IDs and final results are in `NOTES-codex.md`.

## What worked locally

Five paired repetitions, median microseconds:

| B | L | main 0495b16 | native candidate |
|---|---|---:|---:|
| 1 | 1 | 6.224 | 5.664 |
| 1 | 8 | 7.376 | 6.928 |
| 2 | 1 | 7.056 | 6.496 |
| 2 | 8 | 8.224 | 7.760 |

The full local candidate suite passed 16/16. The four native cases above each
passed 12 graph replays with changed QKV. Four additional native cases (pages
1/64, scales 1 and 1/64, L=1/8) each passed three changed-QKV calls under memcheck
and synccheck; both tools reported zero errors. `fresh_routes.py` additionally
changes cache contents, page tables, lengths, positions, and locations each replay.

A broader native merge version was tested on 20 nearby sliding shapes plus four
short DP cases; all passed after the dependency fix below. It did not win the
larger shapes against main's newer `_combine_hb`, so the final dispatch is narrower.
The final H100 aggregate was 1.838x against the harness's fixed SGLang/H200 numbers;
this is not itself a judge measurement.

## Important invalid early measurements

The first native merge (`cuda_merge_v1.py`, `submission_merge_v1.py`) used const /
restrict partial pointers. NVCC hoisted read-only partial loads before
`griddepcontrol.wait` (`ACQBULK` in SASS), despite the inline assembly memory
clobber. Its attractive initial 1.879x aggregate and ~4.9 us decode-1 time are
**INVALID**: graph timing on unchanged inputs hid stale partial reads. Nearby B=48
and repeated eager calls exposed errors/NaNs with PDL on; disabling PDL avoided
it. No version with that bug was checked or submitted to the judge.

The correction uses explicit volatile `ld.global.cg.v2.f32` after the wait. SASS
confirms that coherent partial reads follow ACQBULK (`dependency-order.txt`). The
corrected final candidate additionally passes graph replays with fresh data.
Relevant NVIDIA PTX specification:
https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol

The files containing `merge-v1`, `merge-pairs-v1`, and `integrated-bench.txt` retain
the invalid results solely to document the bug. `merge-valid-pairs.txt`,
`merge-warp-pairs.txt`, `final-pairs.txt`, and `final-bench.txt` use the correction.

## Other experiments, not dispatched

- `cuda_sliding.py`: native decode, one independent attention chain per warp,
  cp.async gather, FP16 MMA QK/PV, normalized/RoPE/FP8 queries and new cache entries.
  Supports 1/2/4 warps and 2/3 stages. Global FP32 partials are either merged by
  Triton or by the last arriving CTA in the same launch. The best two-launch
  results were ~11.1/15.9/20.8 us at B=1/16/32, slower than Triton. Fused merge lost.
  `driver.py quick|sweep` reproduces. The default and all swept configurations
  passed comparisons on the three scored decode sliding shapes.
- `cuda_cluster.py`: replaces global partials/counter with Hopper distributed
  shared memory and cluster barriers. Correct on the swept scored cases, but
  best times ~13.4/16.8/23.5 us. `cluster_driver.py` reproduces.
- `cuda_verify_cluster.py`: WGMMA sliding verify adapted from session 1 with
  producer/consumer warps. **NOT CORRECT / NOT A CANDIDATE**: intermittent NaNs at
  multiple splits. Separate partial storage and CUDA atomic flag access did not
  resolve it. Debugging shows the global partials themselves can be NaN, so the
  issue precedes the cluster reduction. The first racecheck reported hazards in
  the staging pipeline and flags; no clean racecheck was obtained for this path.
  `cuda_verify_debug.py` and `verify_debug_parts.py` preserve the isolation probe.
  Do not infer valid performance or reuse this kernel without fixing and validating.
- `cuda_merge.py`: corrected serial-weight merge, variable warps per CTA.
- `cuda_merge_constants.py`: additionally specializes head count and query length;
  removes runtime divisions but did not materially improve end-to-end timing.
- `cuda_merge_warp.py`: cooperative-weight merge used for the final candidate;
  still exposes warp-count variants for the local sweep.

The full native attention prototypes were not submitted. All caches/partials are
computed inside the timed call; the production candidate only builds its extension
at import time and caches no input-derived values.
