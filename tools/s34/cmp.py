"""Output of plan override B vs A on a scored shape, over repeated calls and a CUDA graph's replays."""
import json, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
shape, pa, pb = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
orig = submission._plan
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != shape:
        continue
    d = bench.data_for(load, c)
    def run(p):
        submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **p}
        try:
            return submission.kernel_fn(d).clone()
        finally:
            submission._plan = orig
    a = run(pa)
    for i in range(5):
        b = run(pb)
        print("call", i, "max abs diff", (a.float() - b.float()).abs().max().item(),
              "nonfinite", (~torch.isfinite(b.float())).sum().item())
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **pb}
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = submission.kernel_fn(d)
    submission._plan = orig
    for i in range(200):
        g.replay()
    torch.cuda.synchronize()
    print("after 200 replays max abs diff", (a.float() - o.float()).abs().max().item())
