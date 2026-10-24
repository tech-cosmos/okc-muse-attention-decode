"""Paired A/B of submission.MERGE1 on a scored shape."""
import os, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
shape = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 9
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == shape:
        a, b = [], []
        for _ in range(rounds):
            submission.MERGE1 = False; a.append(bench.run_shape(load, c, {}))
            submission.MERGE1 = True; b.append(bench.run_shape(load, c, {}))
        d = [y - x for x, y in zip(a, b)]
        print(shape, "old", round(statistics.median(a), 2), "new", round(statistics.median(b), 2),
              "paired", round(statistics.median(d), 2), "wins", sum(x < 0 for x in d), "/", rounds, flush=True)
