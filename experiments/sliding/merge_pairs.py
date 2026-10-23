import sys,json,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding import cuda_merge
orig=bench.submission._combine
for load,c in bench.SCORED:
 if c['layer']!='sliding':continue
 data=bench.data_for(load,c)
 for rep in range(3):
  for w in ((0,2,4,8) if rep%2==0 else (8,4,2,0)):
   bench.submission._combine=orig if w==0 else cuda_merge.Merge(orig,w)
   us=bench.time_graph(lambda:bench.submission.kernel_fn(data))-bench.empty_cost()
   print(json.dumps(dict(load=load,warps=w,rep=rep,us=us)),flush=True)
bench.submission._combine=orig
