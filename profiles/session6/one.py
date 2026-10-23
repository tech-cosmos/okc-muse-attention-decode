import sys, os, json; sys.path.insert(0,'/home/daytona/repo/tools'); sys.path.insert(0,'/home/daytona/repo')
import submission, bench, torch
P=json.loads(sys.argv[2])
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == sys.argv[1]:
        data = bench.data_for(load, c)
        orig = submission._plan
        submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **P}
        for _ in range(3):
            bench._FLUSH = bench._FLUSH if bench._FLUSH is not None else torch.empty(512<<20, dtype=torch.uint8, device='cuda')
            bench._FLUSH.zero_()
            submission.kernel_fn(data); torch.cuda.synchronize()
