"""s106: per-kernel CUDA times (torch profiler, cold L2 per call) for scored shapes on the current submission."""
import sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch
from torch.profiler import profile, ProfilerActivity
for shape in sys.argv[1:]:
    load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
    d = bench.data_for(load, c)
    fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3): submission.kernel_fn(d)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            fl.zero_(); submission.kernel_fn(d)
        torch.cuda.synchronize()
    print("==", shape)
    for e in p.key_averages():
        if e.device_type.name == "CUDA" and "fill" not in e.key and "elementwise" not in e.key.lower():
            print(f"   {e.key[:60]:60s} n={e.count:3d} avg {e.device_time_total / e.count:8.2f} us")
