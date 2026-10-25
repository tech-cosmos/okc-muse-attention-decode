"""s106: paired H100 timing of plan overrides on one scored shape (cold L2, bench.time_graph), plus a correctness check.
usage: pair.py '<shape>' '<json list of plan overrides>' [rounds]"""
import json, sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
shape, plans = sys.argv[1], json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 7
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
data = bench.data_for(load, c)
import os
if os.environ.get("WARM"): bench._FLUSH = torch.empty(8, dtype=torch.uint8, device="cuda")
orig = submission._plan
for p in plans:
    submission._plan = lambda cfg, mp, dev, p=p: {**orig(cfg, mp, dev), **p}
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print(p, "ok" if ok else "FAIL", str(msg)[:120], flush=True)
submission._plan = orig
res = [[] for _ in plans]
for r in range(rounds):
    for i, p in enumerate(plans):
        res[i].append(bench.run_shape(load, c, p))
    print(" ".join(f"{x[-1]:7.2f}" for x in res), flush=True)
import statistics
base = res[0]
for i, p in enumerate(plans):
    d = [a - b for a, b in zip(res[i], base)]
    print(f"{statistics.median(res[i]):7.2f}  d {statistics.median(d):+6.2f} wins {sum(x < 0 for x in d)}/{rounds}  {p}")
