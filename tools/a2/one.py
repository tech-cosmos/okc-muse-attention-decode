"""agent2: run one scored shape a few times (for ncu). python tools/a2/one.py 'verify 32' sliding"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
load, layer = sys.argv[1], sys.argv[2]
for l, c in bench.SCORED:
    if l == load and c["layer"] == layer:
        data = bench.data_for(l, c)
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            flush.zero_()
            submission.kernel_fn(data)
        torch.cuda.synchronize()
