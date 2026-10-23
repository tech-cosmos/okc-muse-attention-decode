import sys,json,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding import cuda_cluster as native
import reference
for load,c in bench.SCORED:
    if c['q_len']!=1 or c['layer']!='sliding':continue
    data=bench.data_for(load,c)
    print('BASE',load,bench.time_graph(lambda:bench.submission.kernel_fn(data))-bench.empty_cost(),flush=True)
    for s in (1,2,4,8):
        for w in (1,2,4):
            out=native.run(data,splits=s,warps=w)
            torch.cuda.synchronize()
            ok,msg=reference.check_implementation(data,reference.collect(data,out))
            print('CHECK',load,s,w,ok,msg,flush=True)
            if not ok:sys.exit(1)
            us=bench.time_graph(lambda:native.run(data,splits=s,warps=w))-bench.empty_cost()
            print(json.dumps(dict(load=load,splits=s,warps=w,us=us)),flush=True)
