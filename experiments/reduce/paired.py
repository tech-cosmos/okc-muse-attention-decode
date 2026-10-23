import sys,json,statistics,fcntl
sys.path[:0]=['.','experiments/reduce']
import torch,reference,submission,native
from tools import bench
orig=submission._combine
results=[]
for batch,length in [(1,1),(1,8),(2,8),(4,1),(8,1),(16,1),(32,1),(8,8),(16,8),(24,8),(32,8),(48,8)]:
    c=dict(batch=batch,ctx=80000,heads='tp8',layer='full',page=1,q_len=length,dist='coherent')
    if batch>1:c['spread']='agentic'
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=reference.generate_input(seed=99,**c)
        ref=submission.kernel_fn(data).clone();torch.cuda.synchronize()
    candidates=[4,8] if batch<=2 else ([2,4,8] if length==1 else [1,2,4])
    for w in candidates:
        vals={'original':[],'native':[]}
        for i in range(5):
            for name in (['original','native'] if i%2==0 else ['native','original']):
                submission._combine=orig if name=='original' else native.Wrapper(w)
                with open('/tmp/okc-gpu.lock','w') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    out=submission.kernel_fn(data);torch.cuda.synchronize()
                    err=(torch.norm(out.float()-ref.float())/torch.norm(ref.float())).item()
                    assert torch.isfinite(out).all() and err<.005,(c,w,err)
                vals[name].append(bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost())
        r=dict(batch=batch,length=length,warps=w,original=statistics.median(vals['original']),native=statistics.median(vals['native']),measurements=vals)
        print(json.dumps(r),flush=True);results.append(r)
    submission._combine=orig
    del data,ref,out;torch.cuda.empty_cache()
open('experiments/reduce/results/paired.json','w').write(json.dumps(results,indent=2)+'\n')
