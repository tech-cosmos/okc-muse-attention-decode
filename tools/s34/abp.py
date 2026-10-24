"""Paired A/B of plan overrides on a scored shape: abp.py '<shape>' rounds '<json A>' '<json B>' [...]
Each round times every override once (interleaved); prints medians and paired deltas vs the first."""
import json, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
shape, rounds = sys.argv[1], int(sys.argv[2])
plans = [json.loads(a) for a in sys.argv[3:]]
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == shape:
        ts = [[] for _ in plans]
        for _ in range(rounds):
            for i, p in enumerate(plans):
                ts[i].append(bench.run_shape(load, c, p))
        for i, p in enumerate(plans):
            d = [y - x for x, y in zip(ts[0], ts[i])]
            print(f"{shape} {round(statistics.median(ts[i]), 2):7} paired {round(statistics.median(d), 2):6} "
                  f"wins {sum(x < 0 for x in d)}/{rounds} {p}", flush=True)
