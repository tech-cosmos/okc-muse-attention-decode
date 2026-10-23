import sys,json,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding import cuda_sliding as native
import reference
orig=bench.submission.kernel_fn
mode=sys.argv[1] if len(sys.argv)>1 else 'quick'
if mode=='test':
    bench.submission.kernel_fn=lambda data:native.run(data) if data['config']['window'] and data['config']['q_len']==1 else orig(data)
    sys.exit(bench.test())
for load,c in bench.SCORED:
    if c['q_len']!=1 or c['layer']!='sliding':continue
    data=bench.data_for(load,c)
    out=native.run(data)
    torch.cuda.synchronize()
    ok,msg=reference.check_implementation(data,reference.collect(data,out))
    print('CHECK',load,ok,msg,flush=True)
    if not ok:sys.exit(1)
    ref=out.clone()
    print('BASE',load,bench.time_graph(lambda:orig(data))-bench.empty_cost(),flush=True)
    plans=[(4,4,3,True),(4,4,3,False)] if mode=='quick' else [(s,w,st,f) for s in (2,4,8,16) for w in (1,2,4) for st in (2,3) for f in (True,False)]
    for s,w,st,f in plans:
        out=native.run(data,splits=s,warps=w,stages=st,fused=f)
        torch.cuda.synchronize()
        err=(out.float()-ref.float()).norm()/ref.float().norm()
        if err>0.03 or not torch.isfinite(out).all():raise RuntimeError((load,s,w,st,f,err))
        us=bench.time_graph(lambda:native.run(data,splits=s,warps=w,stages=st,fused=f))-bench.empty_cost()
        print(json.dumps(dict(load=load,splits=s,warps=w,stages=st,fused=f,us=us,err=float(err))),flush=True)
