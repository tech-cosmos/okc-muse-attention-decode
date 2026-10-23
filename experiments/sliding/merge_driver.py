import sys,json,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding import cuda_merge
import reference
orig=bench.submission._combine
if len(sys.argv)>1 and sys.argv[1]=='test':
    bench.submission._combine=cuda_merge.Merge(orig)
    sys.exit(bench.test())
for load,c in bench.SCORED:
    data=bench.data_for(load,c)
    out=bench.submission.kernel_fn(data)
    torch.cuda.synchronize()
    print('BASE',load,c['layer'],bench.time_graph(lambda:bench.submission.kernel_fn(data))-bench.empty_cost(),flush=True)
    for w in (1,2,4,8):
        bench.submission._combine=cuda_merge.Merge(orig,w)
        out=bench.submission.kernel_fn(data)
        torch.cuda.synchronize()
        ok,msg=reference.check_implementation(data,reference.collect(data,out))
        print('CHECK',load,c['layer'],w,ok,msg,flush=True)
        if not ok:sys.exit(1)
        us=bench.time_graph(lambda:bench.submission.kernel_fn(data))-bench.empty_cost()
        print(json.dumps(dict(load=load,layer=c['layer'],warps=w,us=us)),flush=True)
        bench.submission._combine=orig
