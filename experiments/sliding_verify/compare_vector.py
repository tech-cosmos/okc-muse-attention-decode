"""Balanced AB/BA pairs with cold L2 and complete independent correctness checks."""
import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_verify import vector as native
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
 def capture(fn):
  for _ in range(3):out=fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 def time(g):
  flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
 x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
 for b,ctx,page,heads,scale,dist in [(1,80000,1,'tp8',1.,'soft'),(2,80000,1,'tp8',1.,'soft'),(16,80000,1,'tp8',1.,'soft'),(32,80000,1,'tp8',1.,'soft'),(1,10,64,'tp8',1.,'soft'),(2,4000,64,'tp8',1./64,'edges'),(1,2049,64,'dp',1./64,'coherent')]:
  c=dict(batch=b,ctx=ctx,heads=heads,layer='sliding',page=page,q_len=8,dist=dist,kv_scale=scale)
  if b>1:c['spread']='agentic'
  d=reference.generate_input(seed=102,**c)
  for early in (False,True):
   gs=[]
   for fn in (submission.kernel_fn,lambda d:native.run(d,early=early)):
    out=fn(d);torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(c,early,msg)
    g,out=capture(lambda:fn(d));g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(c,early,msg);gs.append(g)
   raw=[]
   for r in range(40):
    order=[0,1,1,0] if r%2==0 else [1,0,0,1]
    t=[[],[]];empty=time(eg)
    for i in order:t[i].append(time(gs[i])-empty)
    raw.append({'empty':empty,'parent':t[0],'native':t[1]})
   means=[(statistics.mean(t['parent']),statistics.mean(t['native'])) for t in raw]
   rec={'shape':c,'early':early,'parent_us':statistics.median([a for a,b in means]),'native_us':statistics.median([b for a,b in means]),'native_minus_parent_us':statistics.median([b-a for a,b in means]),'parent_over_native':statistics.median([a/b for a,b in means]),'raw':raw}
   results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
   del gs,out,g;torch.cuda.empty_cache()
  del d;torch.cuda.empty_cache()
(Path(__file__).resolve().parent/'results/balanced-vector.json').write_text(json.dumps(results,indent=2)+'\n')
