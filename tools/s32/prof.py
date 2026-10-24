import sys, fcntl, torch; sys.path.insert(0,'/home/daytona/repo/tools'); sys.path.insert(0,'/home/daytona/repo'); import bench, submission
from torch.profiler import profile, ProfilerActivity
for sh in sys.argv[1:]:
  for load,c in bench.SCORED:
    if f"{load} {c['layer']}"==sh:
        d=bench.data_for(load,c); submission.kernel_fn(d); torch.cuda.synchronize()
        flush=torch.empty(256<<20, dtype=torch.uint8, device='cuda')
        with open('/tmp/okc-gpu.lock','w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(20):
                    flush.zero_(); submission.kernel_fn(d)
                torch.cuda.synchronize()
        ev={}
        for e in p.events():
            if e.device_type.name=='CUDA' and 'zero' not in e.name and 'fill' not in e.name.lower():
                ev.setdefault(e.name[:40],[]).append(e.device_time)
        import statistics
        print(sh, {k:(round(statistics.median(v),2),len(v)) for k,v in ev.items()})
