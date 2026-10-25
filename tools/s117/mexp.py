"""s117: merge exposure on verify full (H100, cold): whole call vs the same call with _combine_bal not launched
(timing only, output wrong). usage: mexp.py '<shape>' [rounds]; env WARM=1 = no flush."""
import os, sys, statistics; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch
shape = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
if os.environ.get("WARM"): bench._FLUSH = torch.empty(8, dtype=torch.uint8, device="cuda")
NAME = os.environ.get("K", "_combine_bal"); real = getattr(submission, NAME)
class _Skip:
    def __getitem__(self, g): return lambda *a, **k: None
res = {"whole": [], "nomerge": []}
for r in range(rounds):
    for k in res:
        setattr(submission, NAME, real if k == "whole" else _Skip())
        res[k].append(bench.run_shape(load, c))
    print(" ".join(f"{v[-1]:7.2f}" for v in res.values()), flush=True)
for k, v in res.items(): print(k, f"{statistics.median(v):7.2f}")
