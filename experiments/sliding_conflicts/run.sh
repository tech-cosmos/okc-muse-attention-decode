#!/bin/sh
set -eu
root=experiments/sliding_conflicts
ncu=/usr/local/cuda-13.0/bin/ncu
metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_gmma.sum
/usr/local/cuda-13.0/bin/nvcc -arch=sm_90a -O3 -lineinfo "$root/copy.cu" -o "$root/copy_probe" > "$root/results/copy-build.txt" 2>&1
for source in 0 1 2; do
 for dest in 0 1 2; do
  stem="$root/results/copy-$source-$dest"
  "$ncu" --profile-from-start off --section SourceCounters --metrics "$metrics" --import-source no --page source --csv --print-source sass --export "$stem" --force-overwrite "$root/copy_probe" "$source" "$dest" > "$stem.csv" 2> "$stem.log"
  "$ncu" --import "$stem.ncu-rep" --page raw --csv > "$stem-hardware.csv"
 done
done
"$ncu" --kernel-name regex:sliding --launch-skip 2 --launch-count 1 --section SourceCounters --metrics "$metrics" --import-source no --page source --csv --print-source sass --export "$root/results/base" --force-overwrite /home/daytona/okc/bin/python "$root/profile.py" base early > "$root/results/base-source.csv" 2> "$root/results/base-source.log"
"$ncu" --import "$root/results/base.ncu-rep" --page raw --csv > "$root/results/base-hardware.csv"
