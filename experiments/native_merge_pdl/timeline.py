"""Instrumented PDL scheduling evidence; these durations are not benchmark timings."""
import fcntl
import sys
sys.path.insert(0,'.')
from experiments.native_merge_pdl.compare import MODULES,cases,capture,check,path,torch,reference
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
    items=[]
    for index in (3,13):
        c=cases()[index];data=reference.generate_input(seed=27103,**c)
        for name,m in MODULES.items():
            g,out=capture(lambda:m.kernel_fn(data));g.replay();check(data,out,(index,name))
            items.append((index,name,data,g,out))
        print('CASE',index,path(data),flush=True)
    torch.cuda.synchronize();torch.cuda.profiler.start()
    for repeat in range(3):
        for index,name,data,g,out in items:
            flush.zero_();torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f'case{index}:{name}:repeat{repeat}'):
                g.replay();torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    for index,name,data,g,out in items:check(data,out,(index,name,'after profile'))
    print('PASS 16 output/cache checks, 24 instrumented complete graph replays',flush=True)
