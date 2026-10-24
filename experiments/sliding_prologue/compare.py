import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_prologue import native,direct
from tools.bench import SCORED
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
 cases=[(name,c) for name,c in SCORED if c['q_len']==1 and c['layer']=='sliding']
 cases += [('nearby2',dict(batch=2,ctx=4000,heads='tp8',layer='sliding',page=64,dist='edges'))]
 for name,c in cases:
  data=reference.generate_input(seed=99,**c)
  graphs=[];names=['production','late','early','direct_late','direct_early']
  for variant in names:
   fn=(lambda:submission.kernel_fn(data)) if variant=='production' else (lambda e=variant.endswith('early'),m=direct if variant.startswith('direct') else native:m.run(data,early=e))
   out=fn();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(name,variant,msg)
   g,out=capture(fn);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(name,variant,msg)
   graphs.append(g)
  times=[[] for _ in range(6)];deltas={v:[] for v in names[1:]}
  for r in range(60):
   order=list(range(6));order=order[r%6:]+order[:r%6]
   if r%2:order.reverse()
   row={}
   for i in order:
    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    a.record();(graphs+[eg])[i].replay();z.record();z.synchronize();row[i]=a.elapsed_time(z)*1000;times[i].append(row[i])
   for i,v in enumerate(names[1:],1):deltas[v].append(row[i]-row[0])
  empty=statistics.median(times[5]);rec={'name':name,'shape':c,'us':{v:statistics.median(times[i])-empty for i,v in enumerate(names)},'minus_production_us':{v:statistics.median(t) for v,t in deltas.items()},'early_minus_late_us':statistics.median([a-b for a,b in zip(times[2],times[1])]),'raw':times}
  results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
  del data,graphs,out;torch.cuda.empty_cache()
(ROOT/'results/comparison-direct.json').write_text(json.dumps(results,indent=2)+'\n')
