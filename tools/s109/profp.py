"""s109: per-kernel CUDA times (torch profiler, L2 flushed per call) for one scored shape under a plan override (JSON argv[2])."""
import sys, json; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch
from torch.profiler import profile, ProfilerActivity
shape, plan = sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
d = bench.data_for(load, c)
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
for _ in range(3): submission.kernel_fn(d)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(20):
        fl.zero_(); torch.cuda._sleep(20000); submission.kernel_fn(d)
    torch.cuda.synchronize()
for e in p.key_averages():
    if e.device_type.name == "CUDA" and "fill" not in e.key and "elementwise" not in e.key.lower() and "sleep" not in e.key:
        print(f"   {e.key[:60]:60s} n={e.count:3d} avg {e.device_time_total / e.count:8.2f} us")
