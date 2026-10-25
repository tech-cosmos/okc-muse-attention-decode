"""s121: per-kernel CUDA times (torch profiler, warm L2 unless FLUSH) for verify 1 full main vs nvs S."""
import sys, os; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
from torch.profiler import profile, ProfilerActivity
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "verify 1 full"][0]
data = reference.generate_input(seed=1, **c)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for v in [dict(NVS_FULL1=False)] + [dict(NVS_FULL1=True, NVS_F1_S=int(s)) for s in sys.argv[1:]]:
    for k, x in v.items():
        setattr(submission, k, x)
    for _ in range(3):
        submission.kernel_fn(data)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            if os.environ.get("FLUSH"):
                flush.zero_()
            submission.kernel_fn(data)
        torch.cuda.synchronize()
    print(v)
    for e in p.key_averages():
        if e.count >= 20 and "fill" not in e.key and "elementwise" not in e.key.lower():
            print(f"   {e.key[:60]:60s} {e.device_time:8.2f} us x{e.count}")
