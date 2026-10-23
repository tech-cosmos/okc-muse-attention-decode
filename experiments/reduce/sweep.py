import sys, json, statistics, fcntl
sys.path[:0]=['.','experiments/reduce']
import torch, reference, submission
from tools import bench
import native
orig=submission._combine
results=[]
for batch,length in [(1,1),(1,8),(2,8),(4,1),(16,1),(32,1),(16,8),(32,8)]:
    c=dict(batch=batch,ctx=80000,heads='tp8',layer='full',page=1,q_len=length,dist='coherent')
    if batch>1:c['spread']='agentic'
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=reference.generate_input(seed=99,**c)
        ref=submission.kernel_fn(data).clone();torch.cuda.synchronize()
        ok,msg=reference.check_implementation(data,reference.collect(data,ref));assert ok,msg
    for w in [0,1,2,4,8]:
        plan=submission._plan(data['config'],data['page_table'].shape[1],data['qkv'].device)
        if w and plan['splits']>w*128:continue
        submission._combine=orig if w==0 else native.Wrapper(w)
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            out=submission.kernel_fn(data);torch.cuda.synchronize()
            error=(torch.norm(out.float()-ref.float())/torch.norm(ref.float())).item()
            assert torch.isfinite(out).all() and error<.005,(c,w,error)
        ts=[bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost() for _ in range(3)]
        r=dict(batch=batch,length=length,warps=w,time=statistics.median(ts),times=ts,error=error)
        print(json.dumps(r),flush=True);results.append(r)
    submission._combine=orig
    del data,ref,out;torch.cuda.empty_cache()
open('experiments/reduce/results/sweep.json','w').write(json.dumps(results,indent=2)+'\n')
