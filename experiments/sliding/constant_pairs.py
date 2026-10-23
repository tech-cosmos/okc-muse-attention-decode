import sys,torch,importlib.util,json
sys.path.insert(0,'.')
from tools import bench
import reference,submission
from experiments.sliding import cuda_merge_constants as native
spec=importlib.util.spec_from_file_location('parent','experiments/sliding/submission_main_0495.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
old=submission._NATIVE_MERGE
for load,c in bench.SCORED:
 if c['layer']!='sliding':continue
 data=bench.data_for(load,c)
 submission._NATIVE_MERGE=native.module
 o=submission.kernel_fn(data);torch.cuda.synchronize()
 ok,msg=reference.check_implementation(data,reference.collect(data,o))
 print('CHECK',load,ok,msg,flush=True)
 if not ok:raise RuntimeError(msg)
 for rep in range(3):
  for name in (('parent','old','new') if rep%2==0 else ('new','old','parent')):
   if name=='parent':fn=parent.kernel_fn
   else:
    submission._NATIVE_MERGE=old if name=='old' else native.module;fn=submission.kernel_fn
   us=bench.time_graph(lambda:fn(data))-bench.empty_cost()
   print(json.dumps(dict(load=load,rep=rep,name=name,us=us)),flush=True)
