import sys,torch,importlib.util,json
sys.path.insert(0,'.')
from tools import bench
import reference,submission
from experiments.sliding import cuda_merge as native
spec=importlib.util.spec_from_file_location('parent','experiments/sliding/submission_main_0495.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
class Adapter:
 def __init__(self,w):self.w=w
 def combine(self,qkv,pa,pm,vs,out,hq,hkv,l,s,pdl):return native.module.combine(qkv,pa,pm,vs,out,hq,hkv,l,s,self.w,pdl)
for load,c in bench.SCORED:
 if c['layer']!='sliding':continue
 data=bench.data_for(load,c)
 for rep in range(3):
  for w in ((0,1,2,4,8) if rep%2==0 else (8,4,2,1,0)):
   submission._NATIVE_MERGE=Adapter(w)
   fn=parent.kernel_fn if w==0 else submission.kernel_fn
   us=bench.time_graph(lambda:fn(data))-bench.empty_cost()
   print(json.dumps(dict(load=load,rep=rep,warps=w,us=us)),flush=True)
