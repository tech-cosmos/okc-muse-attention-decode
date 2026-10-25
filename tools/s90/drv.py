"""Run one scored shape's kernel_fn a few times (for ncu)."""
import sys, torch; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench
load, layer = sys.argv[1], sys.argv[2]
load_, c = [x for x in bench.SCORED if x[0] == load and x[1]["layer"] == layer][0]
d = bench.data_for(load_, c)
import os, json
if os.environ.get("PLAN"):
    o = submission._plan; pl = json.loads(os.environ["PLAN"])
    submission._plan = lambda cfg, mp, dev: {**o(cfg, mp, dev), **pl}
for _ in range(4):
    submission.kernel_fn(d)
torch.cuda.synchronize()
