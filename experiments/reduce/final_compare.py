"""Same-process paired comparison of integrated candidate versus prior Triton full merge.
The baseline shim changes only the new combine_full dispatch back to _combine.
"""
import sys,fcntl,json,statistics
sys.path[:0]=['.','experiments/reduce']
import torch,triton,reference,submission
from tools import bench
module=submission._NATIVE_MERGE
class Baseline:
    combine=getattr(module,"combine",None)
    @staticmethod
    def combine_full(qkv,pa,pm,vs,out,hq,hkv,length,splits,pdl):
        submission._combine[(qkv.shape[0]*length,hq)](qkv,pa,pm,vs,out,HQ=hq,HKV=hkv,G=hq//hkv,L=length,M=length*hq//hkv,D=64,S=splits,S_PAD=triton.next_power_of_2(splits),SB=min(128,triton.next_power_of_2(splits)),PDL=pdl,num_warps=4 if splits>32 else 2,launch_pdl=pdl)
results=[]
cases=[(1,1,x,1) for x in [4096,8192,16384,40000,80000,120000]]+[(b,8,x,p) for b,x,p in [(1,4096,64),(1,8192,1),(1,16384,1),(1,40000,1),(1,80000,1),(1,120000,64),(2,80000,1),(3,80000,1)]]
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
    vals={'prior':[],'candidate':[]}
    for i in range(5):
        for name in (['prior','candidate'] if i%2==0 else ['candidate','prior']):
            submission._NATIVE_MERGE=Baseline if name=='prior' else module
            vals[name].append(bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost())
    r=dict(case=c,prior=statistics.median(vals['prior']),candidate=statistics.median(vals['candidate']),error=err,times=vals)
    print(json.dumps(r),flush=True);results.append(r)
    submission._NATIVE_MERGE=module
    del data,ref,out;torch.cuda.empty_cache()
open('experiments/reduce/results/final-paired.json','w').write(json.dumps(results,indent=2)+'\n')
