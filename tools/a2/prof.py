import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
from torch.profiler import profile, ProfilerActivity
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != "verify 32 sliding": continue
    data = bench.data_for(load, c)
    for v in [int(x) for x in sys.argv[1:]]:
        submission.NVS_FUSE = v
        for _ in range(3): submission.kernel_fn(data)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(20): submission.kernel_fn(data)
            torch.cuda.synchronize()
        for e in p.key_averages():
            if e.device_type.name == "CUDA" and e.count >= 20: print(v, e.key[:40], round(e.device_time_total / e.count, 2), e.count)
