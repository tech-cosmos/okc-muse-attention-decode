import sys,json,fcntl
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.direct_v import baseline,direct
records=[]
for batch,ctx,page in [(1,257,1),(1,512,64),(1,80000,1),(16,80000,1),(32,80000,1)]:
 c=dict(batch=batch,ctx=ctx,heads='tp8',layer='full',page=page,q_len=8,dist='coherent')
 if batch>1:c['spread']='agentic'
 d=reference.generate_input(seed=99,**c);base=submission.kernel_fn(d)
 with open('/tmp/okc-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  ok,msg=reference.check_implementation(d,reference.collect(d,base));assert ok,msg
  for stage in (3,4,5):
   opts=dict(splits=max(1,264//batch),stages=stage)
   out=direct.run(d,**opts)
   ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(c,opts,msg)
   err=((out.float()-base.float()).norm()/base.float().norm()).item()
   print('PASS',c,opts,'relerr',err,flush=True)
 if ctx<80000:continue
 for mod in [baseline,direct]:
  for stage in (3,4,5):
   opts=dict(splits=max(1,264//batch),stages=stage)
   t=bench.time_graph(lambda:mod.run(d,**opts))-bench.empty_cost()
   rec=dict(case=c,module=mod.__name__,**opts,us=t)
   print(rec,flush=True);records.append(rec)
 t=bench.time_graph(lambda:submission.kernel_fn(d))-bench.empty_cost()
 print('production',batch,t,flush=True)
 records.append(dict(case=c,module='production',us=t))
 del d,out,base;torch.cuda.empty_cache()
open('experiments/direct_v/results/bench.json','w').write(json.dumps(records,indent=2))
