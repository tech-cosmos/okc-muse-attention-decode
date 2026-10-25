"""s118: paired A/B of module-level switches on scored shapes (bench.run_shape, cold L2).
usage: ab.py '<shape substr>' '<json list of {name: value} dicts>' [rounds]"""
import sys, json, statistics; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
flt, vars_, rounds = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 7
for load, c in bench.SCORED:
    if flt not in f"{load} {c['layer']}":
        continue
    res = [[] for _ in vars_]
    for r in range(rounds):
        for i, v in enumerate(vars_):
            for k, x in v.items():
                setattr(submission, k, x)
            res[i].append(bench.run_shape(load, c))
    base = res[0]
    print(f"== {load} {c['layer']}", flush=True)
    for v, t in zip(vars_, res):
        wins = sum(a < b for a, b in zip(t, base))
        print(f"  {statistics.median(t):8.2f} us  wins vs first {wins}/{rounds}  {v}", flush=True)
