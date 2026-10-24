#!/bin/sh
set -eu
export TORCH_EXTENSIONS_DIR="$PWD/.build/session21"
py=/home/daytona/okc/bin/python
root=experiments/sliding_fp8pairs
for variant in query both; do
    export PAIR_VARIANT="$variant"
    "$py" "$root/validate.py" > "$root/results/validate-$variant.txt" 2>&1
    "$py" "$root/compare.py" > "$root/results/compare-$variant.txt" 2>&1
    "$py" "$root/fresh.py" > "$root/results/fresh-$variant.txt" 2>&1
done
