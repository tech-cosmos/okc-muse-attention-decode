"""s125: verify full intercept. Whole cold time (bench.run_shape) + per-kernel durations (torch profiler,
one call after warmup) for verify full at several batch/ctx. usage: icpt.py [rounds]"""
import sys, statistics; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, torch
from torch.profiler import profile, ProfilerActivity
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
L = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for batch, ctx in [(8, 80000), (16, 40000), (16, 80000), (24, 80000), (32, 40000), (16, 160000), (32, 80000)]:
    c = dict(batch=batch, ctx=ctx, heads="tp8", layer="full", page=1, q_len=L, dist="coherent", spread="agentic")
    load = f"b{batch} c{ctx}"
    ts = [bench.run_shape(load, c) for _ in range(rounds)]
    d = bench.data_for(load, c)
    for _ in range(2): submission.kernel_fn(d)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        submission.kernel_fn(d); torch.cuda.synchronize()
    ev = [(e.name[:22], round(e.device_time, 1)) for e in p.events() if e.device_type.name == "CUDA"]
    print(f"B{batch:3d} ctx{ctx:7d} whole {statistics.median(ts):8.2f}", ev, flush=True)
