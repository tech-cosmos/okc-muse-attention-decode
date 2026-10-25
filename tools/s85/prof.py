"""s85: per-kernel H100 times (torch profiler, cold L2) for main's sliding shapes."""
import sys, torch, fcntl
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
from torch.profiler import profile, ProfilerActivity
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
for load, c in bench.SCORED:
    if c["layer"] != sys.argv[1]: continue
    data = bench.data_for(load, c)
    submission.kernel_fn(data); torch.cuda.synchronize()
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(20):
                torch.sum(flush.view(torch.int64), 0, out=sink)
                submission.kernel_fn(data)
            torch.cuda.synchronize()
    ev = {}
    for e in p.events():
        if e.device_type.name == "CUDA" and "sum" not in e.name and "reduce" not in e.name.lower():
            ev.setdefault(e.name[:40], []).append(e.device_time)
    print(load, {k: round(sorted(v)[len(v) // 2], 2) for k, v in ev.items()}, flush=True)
