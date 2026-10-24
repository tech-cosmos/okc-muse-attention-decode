import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_prologue import production_parent as parent
parent._ROPE=submission._ROPE
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
 cases=[(1,10,1),(1,64,64),(2,1024,64),(1,2047,1),(2,2048,1),(1,2049,64),(2,4000,64),(1,80000,1),(2,80000,1),(1,120000,64),(2,120000,64),(3,4000,1)]
 for b,ctx,page in cases:
  c=dict(batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft')
  data=reference.generate_input(seed=103,**c);graphs=[];names=['production','native']
  for variant in names:
   fn=(lambda:parent.kernel_fn(data)) if variant=='production' else (lambda:submission.kernel_fn(data))
   out=fn();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,variant,msg)
   g,out=capture(fn);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,variant,msg);graphs.append(g)
  times=[[] for _ in range(3)]
  for r in range(60):
   order=list(range(3));order=order[r%3:]+order[:r%3]
   if r%2:order.reverse()
   for i in order:
    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    a.record();(graphs+[eg])[i].replay();z.record();z.synchronize();times[i].append(a.elapsed_time(z)*1000)
  empty=statistics.median(times[2]);rec={'shape':c,'us':{v:statistics.median(times[i])-empty for i,v in enumerate(names)},'native_minus_production_us':statistics.median([a-b for a,b in zip(times[1],times[0])]),'raw':times};results.append(rec)
  print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
  del data,graphs,out;torch.cuda.empty_cache()
(ROOT/'results/integrated-pairs.json').write_text(json.dumps(results,indent=2)+'\n')
