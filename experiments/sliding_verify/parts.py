"""Offline cost diagnostic: omit only merge to time attention; never a submission."""
import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_verify import native
class Noop:
 def __getitem__(self,g):return lambda *a,**kw:None
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
 for b,ctx,page in [(1,80000,1),(1,10,64),(16,80000,1)]:
  c=dict(batch=b,ctx=ctx,page=page,heads='tp8',layer='sliding',q_len=8,dist='soft')
  if b>1:c['spread']='agentic'
  d=reference.generate_input(seed=102,**c);graphs={}
  for label,fn in [('parent',lambda:submission.kernel_fn(d)),('native_late',lambda:native.run(d,early=False)),('native_early',lambda:native.run(d,early=True))]:
   g,out=capture(fn);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg;graphs[label]=g
   old=submission._combine_hb
   submission._combine_hb=Noop()
   try:g,out=capture(fn);graphs[label+'_attention']=g
   finally:submission._combine_hb=old
  values={k:[] for k in graphs}
  for r in range(60):
   empty=time(eg);order=list(graphs);order=order[r%6:]+order[:r%6]
   if r%2:order.reverse()
   for k in order:values[k].append(time(graphs[k])-empty)
  rec={'shape':c,'medians_us':{k:statistics.median(v) for k,v in values.items()},'raw':values};results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
  del graphs,d,g,out;torch.cuda.empty_cache()
(Path(__file__).resolve().parent/'results/parts.json').write_text(json.dumps(results,indent=2)+'\n')
