"""Same-process paired comparison of integrated candidate versus prior Triton full merge.
The baseline shim changes only the new combine_full dispatch back to _combine.
"""
import sys,fcntl,json,statistics
sys.path[:0]=['.','experiments/reduce']
import torch,triton,reference,submission,cooperative
from tools import bench
module=submission._NATIVE_MERGE
Baseline=module
module=cooperative.MOD
results=[]
cases=[(1,1,80000,1),(1,8,80000,1),(2,8,80000,1),(2,1,1024,1),(2,1,1500,64),(2,1,2000,1)]
for batch,length,ctx,page in cases:
    c=dict(batch=batch,ctx=ctx,heads='tp8',layer='full',page=page,q_len=length,dist='coherent')
    if batch>1:c['spread']='agentic'
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=reference.generate_input(seed=99,**c)
        submission._NATIVE_MERGE=Baseline
        ref=submission.kernel_fn(data).clone();torch.cuda.synchronize()
        submission._NATIVE_MERGE=module
        out=submission.kernel_fn(data);torch.cuda.synchronize()
        err=(torch.norm(out.float()-ref.float())/torch.norm(ref.float())).item()
        assert torch.isfinite(out).all() and err<.001,(c,err)
        if length==1 and batch==2:
            ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,msg
    vals={'prior':[],'candidate':[]}
    for i in range(5):
        for name in (['prior','candidate'] if i%2==0 else ['candidate','prior']):
            submission._NATIVE_MERGE=Baseline if name=='prior' else module
            vals[name].append(bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost())
    r=dict(case=c,prior=statistics.median(vals['prior']),candidate=statistics.median(vals['candidate']),error=err,times=vals)
    print(json.dumps(r),flush=True);results.append(r)
    submission._NATIVE_MERGE=module
    del data,ref,out;torch.cuda.empty_cache()
open('experiments/reduce/results/cooperative.json','w').write(json.dumps(results,indent=2)+'\n')
