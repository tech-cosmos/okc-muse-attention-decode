"""s126: per scored shape, kernel timeline inside ONE cold CUDA-graph replay (torch profiler; L2 flushed before).
usage: gtl.py [shape filter substring]"""
import sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch, fcntl
from torch.profiler import profile, ProfilerActivity
flt = sys.argv[1] if len(sys.argv) > 1 else ""
lk = open("/tmp/okc-gpu.lock", "w")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name: continue
    d = bench.data_for(load, c)
    for _ in range(3): submission.kernel_fn(d)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): submission.kernel_fn(d)
    g.replay(); torch.cuda.synchronize()
    fcntl.flock(lk, fcntl.LOCK_EX)
    rows = []
    for r in range(5):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
        ev = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
        t0 = ev[0].time_range.start
        rows.append("  ".join(f"{e.name[:16]}@{e.time_range.start - t0:.1f}+{e.time_range.end - e.time_range.start:.1f}" for e in ev))
    fcntl.flock(lk, fcntl.LOCK_UN)
    print(name + ":\n   " + "\n   ".join(rows), flush=True)
