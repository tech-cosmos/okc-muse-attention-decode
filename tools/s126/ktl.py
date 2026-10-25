"""s126: per scored shape, kernel timeline (start offset, duration) of one warm call via torch profiler (eager, not graph)."""
import sys, json; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch, fcntl
from torch.profiler import profile, ProfilerActivity
lk = open("/tmp/okc-gpu.lock", "w")
for load, c in bench.SCORED:
    d = bench.data_for(load, c)
    for _ in range(3): submission.kernel_fn(d)
    torch.cuda.synchronize()
    fcntl.flock(lk, fcntl.LOCK_EX)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        submission.kernel_fn(d); torch.cuda.synchronize()
    fcntl.flock(lk, fcntl.LOCK_UN)
    ev = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
    t0 = ev[0].time_range.start
    print(f"{load} {c['layer']}: " + "  ".join(f"{e.name[:22]}@{e.time_range.start - t0:.1f}+{e.time_range.end - e.time_range.start:.1f}" for e in ev), flush=True)
