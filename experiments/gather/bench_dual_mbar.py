import sys,json,itertools
sys.path.insert(0,'.')
from tools import bench
import reference,torch,submission
from experiments.gather import dual_mbar as pair
for batch in (16,32):
 c=dict(batch=batch,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent')
 if batch>1:c['spread']='agentic'
 data=reference.generate_input(seed=99,**c)
 baseline=submission.kernel_fn(data)
 print('reference',batch,reference.check_implementation(data,reference.collect(data,baseline)),flush=True)
 results=[]
 for st,ctas in itertools.product((3,4,5),(264,396,528,792)):
  s=max(1,ctas//batch);opts=dict(splits=s,stages=st)
  out=pair.run(data,**opts)
  err=((out.float()-baseline.float()).norm()/baseline.float().norm()).item()
  assert torch.isfinite(out).all() and err<.025,(opts,err)
  t=bench.time_graph(lambda:pair.run(data,**opts))-bench.empty_cost()
  results.append(dict(us=t,err=err,**opts));print(batch,round(t,3),err,opts,flush=True)
 print('triton',batch,bench.time_graph(lambda:submission.kernel_fn(data))-bench.empty_cost(),flush=True)
 open(f'experiments/gather/results/dual-mbar-{batch}.json','w').write(json.dumps(results,indent=2))
 del data;torch.cuda.empty_cache()
