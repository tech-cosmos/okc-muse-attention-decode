"""List the CUDA-graph nodes each scored shape's kernel_fn captures (kernels, memsets, memcpys)."""
import sys, torch, ctypes
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import bench, submission
from torch.profiler import profile, ProfilerActivity
for load, c in bench.SCORED:
    d = bench.data_for(load, c)
    for _ in range(3): submission.kernel_fn(d)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        submission.kernel_fn(d); torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type.name == "CUDA"]
    print(load, c["layer"], [ (e.name[:40], round(e.device_time,1)) for e in ev])
