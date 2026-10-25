import sys, torch
sys.path.insert(0, "/home/daytona/repo")
import reference, submission
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
cfg = d["config"] if "config" in d else None
out = submission.kernel_fn(d); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    o2 = submission.kernel_fn(d)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
for e in p.key_averages():
    if e.device_time_total > 0: print(f"{e.key[:40]:40s} {e.device_time_total/e.count:9.1f} us x{e.count}")
