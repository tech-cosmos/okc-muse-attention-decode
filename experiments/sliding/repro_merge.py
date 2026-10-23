import sys,torch,importlib.util
sys.path.insert(0,'.')
from tools import bench
import reference,submission
spec=importlib.util.spec_from_file_location('parent','experiments/sliding/submission_parent.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
c=dict(batch=48,ctx=80000,heads='tp8',layer='sliding',page=1,q_len=1,spread='agentic',dist='soft')
data=reference.generate_input(seed=817,**c)
for pdl in (True,False):
 submission.USE_PDL=pdl;parent.USE_PDL=pdl
 for name,f in (('parent',parent.kernel_fn),('new',submission.kernel_fn)):
  for i in range(3):
   o=f(data);torch.cuda.synchronize()
   ok,msg=reference.check_implementation(data,reference.collect(data,o))
   print(name,pdl,i,ok,msg,flush=True)
   if not ok:print('badrows',o.isnan().reshape(48,16,64).any(-1).nonzero().tolist(),flush=True)
