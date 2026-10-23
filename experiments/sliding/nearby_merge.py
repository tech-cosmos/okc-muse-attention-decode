import sys,torch,json,importlib.util
sys.path.insert(0,'.')
from tools import bench
import reference,submission
spec=importlib.util.spec_from_file_location('parent','experiments/sliding/submission_parent.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
for length in (1,8):
 for batch in (2,4,8,12,20,24,48,64):
  for ctx in ((40000,120000) if batch in (8,24) else (80000,)):
   c=dict(batch=batch,ctx=ctx,heads='tp8',layer='sliding',page=1,q_len=length,spread='agentic',dist='soft')
   data=reference.generate_input(seed=817,**c)
   out=submission.kernel_fn(data);torch.cuda.synchronize()
   ok,msg=reference.check_implementation(data,reference.collect(data,out))
   if not ok:raise RuntimeError((c,msg))
   native=bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost()
   old=bench.time_graph(lambda:parent.kernel_fn(data))-bench.empty_cost()
   print(json.dumps(dict(**c,passed=ok,native=native,parent=old)),flush=True)
   del data,out;torch.cuda.empty_cache()
# Short contexts exercise both supported and fallback split counts; DP varies the head count.
for ctx,b,l,page,scale in ((1,1,1,1,1),(100,3,8,64,1),(257,4,1,1,1/64),(1024,2,8,64,1/64)):
 c=dict(batch=b,ctx=ctx,heads='dp',layer='sliding',page=page,q_len=l,kv_scale=scale,dist='soft')
 data=reference.generate_input(seed=3,**c);out=submission.kernel_fn(data);torch.cuda.synchronize()
 ok,msg=reference.check_implementation(data,reference.collect(data,out))
 print('EXTRA',c,ok,msg,flush=True)
 if not ok:raise RuntimeError(msg)
