import sys,torch,json,importlib.util
sys.path.insert(0,'.')
from tools import bench
import reference,submission
spec=importlib.util.spec_from_file_location('parent','experiments/sliding/submission_main_0495.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
for b in (1,2):
 for l in (1,8):
  c=dict(batch=b,ctx=80000,heads='tp8',layer='sliding',page=1,q_len=l,spread='agentic' if b>1 else 'fixed',dist='soft')
  data=reference.generate_input(seed=99,**c)
  out=submission.kernel_fn(data);torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  print('CHECK',b,l,ok,msg,flush=True)
  if not ok:raise RuntimeError(msg)
  for rep in range(5):
   for name in (('parent','native') if rep%2==0 else ('native','parent')):
    fn=parent.kernel_fn if name=='parent' else submission.kernel_fn
    us=bench.time_graph(lambda:fn(data))-bench.empty_cost()
    print(json.dumps(dict(batch=b,q_len=l,name=name,rep=rep,us=us)),flush=True)
  # Replay after changing inputs so reusing a prior replay's partials cannot pass.
  g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   with torch.cuda.graph(g,stream=stream):out=submission.kernel_fn(data)
  torch.cuda.current_stream().wait_stream(stream)
  for rep in range(12):
   data['qkv'].normal_();g.replay();torch.cuda.synchronize()
   ok,msg=reference.check_implementation(data,reference.collect(data,out))
   if not ok:raise RuntimeError((b,l,rep,msg))
  print('FRESH GRAPH PASS',b,l,12,flush=True)
