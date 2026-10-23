import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch
from tools import bench
from experiments.pack import pack

def validate(d,out):
 p=d['config']['page_size'];pt=d['page_table'];h=d['config']['num_kv_heads']
 for b,n in enumerate(d['seq_lens'].tolist()):
  rr=torch.arange(n,device='cuda'); idx=pt[b,rr//p].long()*p+rr%p
  for src,dst in zip((d['k_cache'],d['v_cache']),out):
   ref=src.view(-1,h,64).view(torch.uint8)[idx]
   assert torch.equal(ref,dst[b,:n].view(torch.uint8))

results=[]
for B in (1,16,32):
 c=dict(batch=B,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent')
 if B>1:c['spread']='agentic'
 d=bench.reference.generate_input(seed=99,**c);out=pack.allocate(d)
 base=bench.time_graph(lambda:bench.submission.kernel_fn(d))-bench.empty_cost()
 rs=[]
 for vec in (16,8,4):
  for unroll in (1,2,4,8):
   for total in (264,528,1056):
    blocks=max(1,(total+B-1)//B)
    fn=lambda:pack.run(d,out,blocks,unroll,vec)
    with open('/tmp/okc-gpu.lock','w') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX);fn();torch.cuda.synchronize();validate(d,out)
    t=bench.time_graph(fn,reps=25)-bench.empty_cost()
    rs.append(dict(vec=vec,unroll=unroll,blocks=blocks,us=t))
 rs.sort(key=lambda r:r['us'])
 record=dict(case=c,production_us=base,bytes=int(d['seq_lens'].sum())*256,results=rs)
 results.append(record)
 print(json.dumps(record),flush=True)
 del d,out;torch.cuda.empty_cache()
Path('experiments/pack/results/sweep.json').write_text(json.dumps(results,indent=2))
