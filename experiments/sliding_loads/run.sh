#!/bin/sh
set -eu
export CUDA_HOME=/usr/local/cuda-13.0
for variant in qpre plain rope; do
 export LOAD_VARIANT="$variant"
 /home/daytona/okc/bin/python -u experiments/sliding_loads/validate.py > "experiments/sliding_loads/results/test-$variant.txt" 2>&1
 /home/daytona/okc/bin/python -u experiments/sliding_loads/bench.py > "experiments/sliding_loads/results/bench-$variant.txt" 2>&1
 /home/daytona/okc/bin/python -u experiments/sliding_loads/compare.py > "experiments/sliding_loads/results/compare-$variant.txt" 2>&1
 done
