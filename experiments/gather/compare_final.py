import sys,json,statistics
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.gather import pair_mbar,gather,gather_bulk
results=[]
for b in (1,8,16,24,32):
 c=dict(batch=b,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent')
 if b>1:c['spread']='agentic'
 data=reference.generate_input(seed=99,**c)
 native=lambda:pair_mbar.run(data,splits=max(1,264//b),stages=4)
 base=lambda:submission.kernel_fn(data)
 ref=base();actual=native()
 assert ((actual.float()-ref.float()).norm()/ref.float().norm()).item()<.025
 tbase=[];tnative=[]
 for rep in range(3):
  order=[('base',base),('native',native)]
  if rep%2:order.reverse()
  times={name:bench.time_graph(fn)-bench.empty_cost() for name,fn in order}
  tbase.append(times['base']);tnative.append(times['native'])
 rec=dict(batch=b,base=tbase,native=tnative,median_base=statistics.median(tbase),median_native=statistics.median(tnative))
 print(rec,flush=True);results.append(rec)
 del data,ref,actual;torch.cuda.empty_cache()
open('experiments/gather/results/final-comparison.json','w').write(json.dumps(results,indent=2))
