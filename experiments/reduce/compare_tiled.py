import sys,json,statistics,fcntl
sys.path[:0]=['.','experiments/reduce']
import torch,reference,submission,native,tiled
from tools import bench
orig=submission._combine
variants={'original':orig}
for w in [4,8]:variants[f'warp_w{w}']=native.Wrapper(w)
for lanes in [8,16]:
    for w in [2,4,8]:variants[f'tile{lanes}_w{w}']=tiled.Wrapper(w,lanes)
results=[]
for batch,length in [(1,1),(1,8),(2,8),(16,1),(16,8),(32,8)]:
    c=dict(batch=batch,ctx=80000,heads='tp8',layer='full',page=1,q_len=length,dist='coherent')
    if batch>1:c['spread']='agentic'
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=reference.generate_input(seed=99,**c)
        ref=submission.kernel_fn(data).clone();torch.cuda.synchronize()
        ok,msg=reference.check_implementation(data,reference.collect(data,ref));assert ok,msg
    for name,wrapper in variants.items():
        submission._combine=wrapper
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            out=submission.kernel_fn(data);torch.cuda.synchronize()
            err=(torch.norm(out.float()-ref.float())/torch.norm(ref.float())).item()
            assert torch.isfinite(out).all() and err<.005,(c,name,err)
        ts=[bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost() for _ in range(3)]
        r=dict(batch=batch,length=length,variant=name,time=statistics.median(ts),times=ts,error=err)
        print(json.dumps(r),flush=True);results.append(r)
    submission._combine=orig
    del data,ref,out;torch.cuda.empty_cache()
open('experiments/reduce/results/tiled.json','w').write(json.dumps(results,indent=2)+'\n')
