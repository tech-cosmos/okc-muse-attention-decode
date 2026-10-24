import argparse,fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.verify_merge import candidate
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=60);p.add_argument('--seed',type=int,default=103);p.add_argument('--variant',choices=['warp','striped'],required=True);p.add_argument('--serial',action='store_true');p.add_argument('--nearby',action='store_true');a=p.parse_args()
cases=[dict(batch=b,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent',spread='agentic') for b in (17,24,32,48,64)]
if a.nearby:
 cases=[dict(batch=b,ctx=ctx,heads=heads,layer='full',page=page,q_len=8,dist=dist,spread='mixed',kv_scale=scale) for b,ctx,heads,page,dist,scale in [(17,4096,'tp8',64,'soft',1.),(24,40000,'tp8',64,'coherent',1/64),(32,120000,'tp8',1,'coherent',1.),(3,6000,'dp',64,'edges',1.),(8,10000,'dp',1,'soft',1/64)]]
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
 def capture(f):
    for _ in range(3):out=f()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=f()
    g.replay();torch.cuda.synchronize();return g,out
 x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
 def time(g):
    flush.zero_();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record();g.replay();e.record();e.synchronize();return s.elapsed_time(e)*1000
 def check(data,out):
    ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,msg
 for c in cases:
  data=reference.generate_input(seed=a.seed,**c);gs=[];outs=[]
  for f in (lambda:submission.kernel_fn(data),lambda:candidate.run(data,a.variant=='striped',not a.serial)):
   out=f();torch.cuda.synchronize();check(data,out)
   g,out=capture(f);check(data,out);gs.append(g);outs.append(out)
  raw=[]
  for r in range(a.rounds):
   vals={j:time(gs[j]) for j in ([0,1] if r%2==0 else [1,0])};raw.append([vals[0],vals[1],time(eg)])
  for g,out in zip(gs,outs):g.replay();torch.cuda.synchronize();check(data,out)
  empty=statistics.median(r[2] for r in raw)
  rec=dict(shape=c,variant=a.variant,serial=a.serial,parent_us=statistics.median(r[0] for r in raw)-empty,native_us=statistics.median(r[1] for r in raw)-empty,paired_delta_us=statistics.median(r[1]-r[0] for r in raw),raw=raw)
  results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
  del gs,outs,g,out,data;torch.cuda.empty_cache()
  (ROOT/f'results/pairs-{a.variant}-{a.serial}-{a.seed}-{a.nearby}.json').write_text(json.dumps(results,indent=2)+'\n')
