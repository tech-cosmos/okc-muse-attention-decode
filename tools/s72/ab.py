"""Paired H100 timing of decode sliding: submission with CSL_MAX_BATCH 0 (Triton) vs N. python tools/s72/ab.py N rounds"""
import sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
n, rounds = int(sys.argv[1]), int(sys.argv[2])
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1:
        continue
    d = bench.data_for(load, c)
    a, b = [], []
    for _ in range(rounds):
        for mb, lst in ((0, a), (n, b)):
            submission.CSL_MAX_BATCH = mb
            lst.append(bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost())
    wins = sum(y < x for x, y in zip(a, b))
    print(f"{load} sliding: triton {sorted(a)[len(a)//2]:.2f}  csl {sorted(b)[len(b)//2]:.2f}  csl wins {wins}/{rounds}", flush=True)
