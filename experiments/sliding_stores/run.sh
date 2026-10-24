#!/bin/sh
set -eu
export TORCH_EXTENSIONS_DIR="$PWD/experiments/sliding_stores/.build-production"
for variant in vector explicit; do
 export STORE_VARIANT="$variant"
 /home/daytona/okc/bin/python -u experiments/sliding_stores/validate.py > "experiments/sliding_stores/results/test-$variant.txt" 2>&1
 /home/daytona/okc/bin/python -u experiments/sliding_stores/bench.py > "experiments/sliding_stores/results/bench-$variant.txt" 2>&1
 /home/daytona/okc/bin/python -u experiments/sliding_stores/compare.py > "experiments/sliding_stores/results/compare-$variant.txt" 2>&1
 /home/daytona/okc/bin/python -u experiments/sliding_stores/fresh.py > "experiments/sliding_stores/results/fresh-$variant.txt" 2>&1
done
