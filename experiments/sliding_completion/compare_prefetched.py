import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_completion import prefetched as native
ROOT=Path(__file__).resolve().parent
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
 def capture(fn):
  for _ in range(3):out=fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
 def time(g):
  flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
  a.record();g.replay();z.record();z.synchronize();return a.elapsed_time(z)*1000
 cases=[(1,80000,1),(2,80000,1),(1,4000,64),(2,1024,64),(1,10,1),(3,4000,1),(16,80000,1),(32,80000,1)]
 for b,ctx,page in cases:
  c=dict(batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft')
  if b>1:c['spread']='agentic'
  data=reference.generate_input(seed=103,**c)
  for label,fn in [('fused',lambda:native.kernel_fn(data)),('control',lambda:native.kernel_fn(data,False))]:
   graphs=[]
   for f in (lambda:submission.kernel_fn(data),fn):
    out=f();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,label,msg)
    g,out=capture(f);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,label,msg);graphs.append(g)
   raw=[]
   for pair in range(40):
    order=[0,1] if pair%2==0 else [1,0]
    vals={i:time(graphs[i]) for i in order};raw.append([vals[0],vals[1],time(eg)])
   empty=statistics.median(r[2] for r in raw)
   rec={'shape':c,'variant':label,'parent_us':statistics.median(r[0] for r in raw)-empty,'candidate_us':statistics.median(r[1] for r in raw)-empty,'paired_delta_us':statistics.median(r[1]-r[0] for r in raw),'raw':raw};results.append(rec)
   print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
   del graphs,out
  del data;torch.cuda.empty_cache()
(ROOT/'results/comparison-prefetched.json').write_text(json.dumps(results,indent=2)+'\n')
