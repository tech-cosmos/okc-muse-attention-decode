"""Compare actual register-pool instructions, static controls, original, and production."""
import sys,json,functools,fcntl
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.register_pool import pool2,static2,late2,late3,baseline
results=[]
for load,c in bench.SCORED:
 if c['q_len']!=8 or c['layer']!='full':continue
 data=reference.generate_input(seed=99,**c)
 variants=[('production',lambda:submission.kernel_fn(data)),('native_baseline',lambda:baseline.run(data,stages=4))]
 variants += [(f'{name}_s4',functools.partial(mod.run,data,stages=4)) for name,mod in [('pool2',pool2),('static2',static2),('late2',late2),('late3',late3)]]
 for name,fn in variants:
  with open('/tmp/okc-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   print('START',load,name,flush=True)
   out=fn();torch.cuda.synchronize()
   ok,msg=reference.check_implementation(data,reference.collect(data,out))
   assert ok,(load,name,msg)
  t=bench.time_graph(fn,reps=60)-bench.empty_cost()
  row=dict(shape=load,variant=name,us=t,reference=ok)
  results.append(row);print(json.dumps(row),flush=True)
 del data,out,variants;torch.cuda.empty_cache()
Path('experiments/register_pool/compare-pools.json').write_text(json.dumps(results,indent=2)+'\n')
