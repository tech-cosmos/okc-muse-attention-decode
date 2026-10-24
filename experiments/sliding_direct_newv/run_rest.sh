#!/bin/sh
set -eu
py=/home/daytona/okc/bin/python
root=experiments/sliding_direct_newv
NEWV_VARIANT=conditional "$py" "$root/compare.py" > "$root/results/compare-conditional.txt" 2>&1
for variant in direct conditional; do
    NEWV_VARIANT="$variant" "$py" "$root/fresh.py" > "$root/results/fresh-$variant.txt" 2>&1
done
"$py" "$root/bitexact.py" > "$root/results/bitexact.txt" 2>&1
for variant in direct conditional; do
    for tool in memcheck synccheck racecheck; do
        NEWV_VARIANT="$variant" /usr/local/cuda-13.0/bin/compute-sanitizer --tool "$tool" --error-exitcode 1 "$py" "$root/sanitize.py" > "$root/results/$tool-$variant.txt" 2>&1
    done
done
for variant in base direct conditional; do
    /usr/local/cuda-13.0/bin/cuobjdump --dump-sass "$root/.build-$variant/codex_sliding_direct_newv_$variant.so" > "$root/results/$variant.sass"
    /usr/local/cuda-13.0/bin/cuobjdump --dump-resource-usage "$root/.build-$variant/codex_sliding_direct_newv_$variant.so" > "$root/results/$variant.resources.txt"
done
