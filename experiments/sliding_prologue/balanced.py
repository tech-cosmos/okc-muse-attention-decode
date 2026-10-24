"""Final end-to-end adjacent AB/BA comparison; cold L2 before every replay."""
import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_prologue import production_parent as parent
parent._ROPE=submission._ROPE
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
 for b,ctx,page in [(1,80000,1),(2,80000,1),(1,4000,64),(2,4000,64),(1,1024,64)]:
  c=dict(batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft')
  d=reference.generate_input(seed=102,**c);gs=[]
  for mod in (parent,submission):
   out=mod.kernel_fn(d);torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg
   g,out=capture(lambda:mod.kernel_fn(d));g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg;gs.append(g)
  raw=[]
  for r in range(60):
   # Alternate which variant starts each balanced adjacent pair.
   order=[0,1,1,0] if r%2==0 else [1,0,0,1]
   t=[[],[]];empty=time(eg)
   for i in order:t[i].append(time(gs[i])-empty)
   raw.append({'empty':empty,'parent':t[0],'native':t[1]})
  means=[(statistics.mean(t['parent']),statistics.mean(t['native'])) for t in raw]
  rec={'shape':c,'parent_us':statistics.median([a for a,b in means]),'native_us':statistics.median([b for a,b in means]),'native_minus_parent_us':statistics.median([b-a for a,b in means]),'parent_over_native':statistics.median([a/b for a,b in means]),'raw':raw}
  results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
  del d,gs,out;torch.cuda.empty_cache()
(Path(__file__).resolve().parent/'results/balanced.json').write_text(json.dumps(results,indent=2)+'\n')
