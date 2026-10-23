import sys,torch
sys.path.insert(0,'.')
import reference
from experiments.sliding import cuda_verify_cluster as native
cases=[dict(batch=b,ctx=ctx,heads=heads,layer='sliding',q_len=8,page=p,kv_scale=sc,spread='mixed',dist=dist) for b,ctx,heads,p,sc,dist in [(1,10,'tp8',1,1,'normal'),(2,2047,'tp8',64,1/64,'edges'),(5,5000,'tp8',1,1,'normal'),(3,4000,'tp8',1,1/64,'soft'),(4,6000,'tp8',1,1,'soft'),(2,6000,'dp',64,1,'soft')]]
for i,c in enumerate(cases):
 data=reference.generate_input(seed=891+i,**c)
 for s in (4,8):
  out=native.run(data,splits=s,stages=3);torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  print('CHECK',c,s,ok,msg,flush=True)
  if not ok:raise RuntimeError(msg)
