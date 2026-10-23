"""Pre-captured graphs, alternating orders, shared lock for all GPU work.
Each graph replay has a cold L2; report paired medians minus empty replay.
"""
import sys,fcntl,json,statistics,gc
sys.path[:0]=['.','experiments/reduce/session5']
import torch,submission,reference,pregate
from tools import bench
mods={'production':submission._NATIVE_MERGE,**pregate.mods}
cases=[(1,1,80000,1),(1,8,80000,1),(2,8,80000,1),(1,1,40000,64),(1,8,120000,64),(2,1,1500,64)]
results=[]
for b,l,ctx,page in cases:
 c=dict(batch=b,q_len=l,ctx=ctx,page=page,heads='tp8',layer='full',dist='coherent')
 if b>1:c['spread']='agentic'
 with open('/tmp/okc-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  data=reference.generate_input(seed=99,**c)
  graphs={};outputs={}
  for name,mod in mods.items():
   submission._NATIVE_MERGE=mod
   outputs[name]=submission.kernel_fn(data)
   torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):outputs[name]=submission.kernel_fn(data)
   graphs[name]=graph
   graph.replay()
  torch.cuda.synchronize()
  ref=outputs['production'].float()
  errors={name:(torch.norm(out.float()-ref)/torch.norm(ref)).item() for name,out in outputs.items()}
  assert all(torch.isfinite(out).all() for out in outputs.values()) and max(errors.values())<.001,errors
  for name in mods:
   ok,msg=reference.check_implementation(data,reference.collect(data,outputs[name]));assert ok,(c,name,msg)
  flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
  z=torch.zeros(1,device='cuda');empty=torch.cuda.CUDAGraph()
  with torch.cuda.graph(empty):z.add_(1)
  graphs['empty']=empty
  times={name:[] for name in graphs}
  events=[]
  names=list(graphs)
  # Rotate which variant goes first and reverse alternate rounds.
  for i in range(100):
   order=names[i%len(names):]+names[:i%len(names)]
   if i%2:order=order[::-1]
   for name in order:
    flush.zero_();a=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    a.record();graphs[name].replay();e.record();events.append((name,a,e))
  torch.cuda.synchronize()
  for name,a,e in events:times[name].append(a.elapsed_time(e)*1000)
  overhead=statistics.median(times['empty'])
  medians={name:statistics.median(ts)-overhead for name,ts in times.items() if name!='empty'}
  diffs={name:statistics.median([a-b for a,b in zip(times[name],times['production'])]) for name in mods if name!='production'}
  result=dict(case=c,medians=medians,paired_delta=diffs,errors=errors,times=times)
  print(json.dumps({k:v for k,v in result.items() if k!='times'}),flush=True);results.append(result)
  del data,outputs,graphs,ref,flush,z,empty,events,a,e,graph
  gc.collect();torch.cuda.empty_cache()
submission._NATIVE_MERGE=mods['production']
from pathlib import Path
Path(sys.argv[1] if len(sys.argv)>1 else 'experiments/reduce/session5/pregate-paired.json').write_text(json.dumps(results,indent=2)+'\n')
