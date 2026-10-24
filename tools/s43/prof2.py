"""Kernel GPU times (torch profiler, flushed) for a shape with a plan override: prof2.py "<shape>" '<json>'"""
import sys, json, fcntl, statistics, torch
sys.path.insert(0, '/home/daytona/repo/tools'); sys.path.insert(0, '/home/daytona/repo')
import bench, submission
from torch.profiler import profile, ProfilerActivity
ov = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **ov}
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != sys.argv[1]: continue
    d = bench.data_for(load, c); submission.kernel_fn(d); torch.cuda.synchronize()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(20):
                flush.zero_(); submission.kernel_fn(d)
            torch.cuda.synchronize()
    ev = {}
    for e in p.events():
        if e.device_type.name == 'CUDA' and 'zero' not in e.name and 'fill' not in e.name.lower():
            ev.setdefault(e.name[:40], []).append(e.device_time)
    print(sys.argv[1], ov, {k: (round(statistics.median(v), 2), len(v)) for k, v in ev.items()})
