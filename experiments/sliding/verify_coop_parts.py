import sys,torch,triton
sys.path.insert(0,'.')
import reference,submission
from experiments.sliding import cuda_verify_coop as native
c=dict(batch=1,ctx=80000,heads='tp8',layer='sliding',page=1,q_len=8,dist='soft')
data=reference.generate_input(seed=99,**c)
for s in (1,2,4,8):
 for st in (2,3):
  o,pa,pm=native.run(data,splits=s,stages=st)
  other=torch.empty_like(o)
  submission._combine[(8,16)](data['qkv'],pa,pm,data['v_scale'],other,HQ=16,HKV=2,G=8,L=8,M=64,D=64,S=s,S_PAD=s,SB=s,PDL=False,num_warps=2)
  torch.cuda.synchronize()
  print(s,st,'out',o.isfinite().all().item(),'pa',pa.isfinite().all().item(),'pm',pm.isfinite().all().item(),'other',other.isfinite().all().item(), 'err', ((o.float()-other.float()).norm()/other.float().norm()).item(),flush=True)
  print('ml ranges',pm.flatten().amin().item(),pm.flatten().amax().item(),flush=True)
