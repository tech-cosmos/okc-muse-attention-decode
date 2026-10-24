import sys,fcntl,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,reference
from experiments.deferred_denominator import native
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for i,c in enumerate([dict(batch=1,ctx=n,heads='tp8',layer='full',page=p) for p in [1,64] for n in [0,1,10,63,64,65,257,4096]]):
  data=reference.generate_input(seed=100+i,**c)
  out=native.kernel_fn(data);torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  print(json.dumps(dict(case=c,ok=ok,message=msg)),flush=True)
  if not ok:break
