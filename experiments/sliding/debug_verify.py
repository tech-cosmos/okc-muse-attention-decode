import sys,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding import cuda_verify_cluster as native
import reference
c=dict(batch=1,ctx=6000,heads='tp8',layer='sliding',page=1,q_len=8,dist='soft')
data=reference.generate_input(seed=99,**c)
for s in (1,2,4,8):
 for st in (2,3):
  o=native.run(data,splits=s,stages=st)
  torch.cuda.synchronize()
  print(s,st,'finite',o.isfinite().all().item(),'nan rows',o.isnan().reshape(8,16,64).any(-1).nonzero().tolist(),flush=True)
