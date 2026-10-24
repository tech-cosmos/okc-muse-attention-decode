#!/bin/sh
set -eu
export CUDA_HOME=/usr/local/cuda-13.0
export LOAD_VARIANT=ordered
/home/daytona/okc/bin/python -u experiments/sliding_loads/validate.py > experiments/sliding_loads/results/test-ordered.txt 2>&1
/home/daytona/okc/bin/python -u experiments/sliding_loads/bench.py > experiments/sliding_loads/results/bench-ordered.txt 2>&1
/home/daytona/okc/bin/python -u experiments/sliding_loads/compare.py > experiments/sliding_loads/results/compare-ordered.txt 2>&1
for variant in qpre plain rope ordered; do
 export LOAD_VARIANT="$variant"
 /home/daytona/okc/bin/python -u experiments/sliding_loads/fresh.py > "experiments/sliding_loads/results/fresh-$variant.txt" 2>&1
 done
for variant in rope ordered; do
 export LOAD_VARIANT="$variant"
 for tool in memcheck synccheck racecheck; do
  /usr/local/cuda-13.0/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 /home/daytona/okc/bin/python -u experiments/sliding_loads/sanitize.py > "experiments/sliding_loads/results/$tool-$variant.txt" 2>&1
 done
 done
for variant in qpre ordered; do
 /usr/local/cuda-13.0/bin/ncu --kernel-name regex:sliding --launch-skip 2 --launch-count 1 --section SourceCounters --warp-sampling-interval 0 --import-source no --page source --csv --print-source sass /home/daytona/okc/bin/python experiments/sliding_loads/profile.py "$variant" early > "experiments/sliding_loads/results/profile-$variant.csv" 2> "experiments/sliding_loads/results/profile-$variant.log"
 done
