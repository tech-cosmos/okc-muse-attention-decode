"""Cold-L2 graph comparison; each swept candidate checked against independent reference."""
import sys,json,functools,fcntl
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.producer_convert import producer,baseline
results=[]
for load,c in bench.SCORED:
 if c['q_len']!=8 or c['layer']!='full':continue
 data=reference.generate_input(seed=99,**c)
 variants=[('production',lambda:submission.kernel_fn(data)),('native_baseline',lambda:baseline.run(data,stages=4))]
 variants += [(f'producer{p}_s{s}',functools.partial(producer.run,data,stages=s,producers=p)) for p in (2,4) for s in (2,3,4,5)]
 for name,fn in variants:
  with open('/tmp/okc-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   out=fn();torch.cuda.synchronize()
   ok,msg=reference.check_implementation(data,reference.collect(data,out))
   assert ok,(load,name,msg)
  t=bench.time_graph(fn,reps=60)-bench.empty_cost()
  row=dict(shape=load,variant=name,us=t,reference=ok)
  results.append(row);print(json.dumps(row),flush=True)
 del data,out,variants;torch.cuda.empty_cache()
Path('experiments/producer_convert/compare.json').write_text(json.dumps(results,indent=2)+'\n')
