"""Reference checks then rotating/reversing-order cold-L2 CUDA graph comparisons."""
import sys,json,fcntl,statistics
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.pv_warpgroup import baseline,swizzle,swizzle_overlap
results=[]
for load,c in bench.SCORED:
    if c['q_len']!=8 or c['layer']!='full': continue
    data=reference.generate_input(seed=99,**c)
    funcs=[('production',lambda:submission.kernel_fn(data))]
    funcs += [(name,lambda mod=mod:mod.run(data,stages=4)) for name,mod in [('baseline',baseline),('swizzle',swizzle),('swizzle_overlap',swizzle_overlap)]]
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        graphs=[]
        for name,fn in funcs:
            out=fn(); torch.cuda.synchronize()
            ok,msg=reference.check_implementation(data,reference.collect(data,out))
            print('REFERENCE',load,name,ok,msg[:100],flush=True);assert ok,msg
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            g=torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                fn()
                with torch.cuda.graph(g,stream=stream): out=fn()
            torch.cuda.current_stream().wait_stream(stream)
            g.replay();torch.cuda.synchronize()
            ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,msg
            graphs.append(g)
        flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
        trivial=torch.zeros(1,device='cuda');g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):trivial.add_(1)
        graphs.append(g)
        times=[[] for _ in graphs]
        for r in range(60):
            order=list(range(len(graphs)));order=order[r%len(order):]+order[:r%len(order)]
            if r%2:order.reverse()
            for i in order:
                flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();graphs[i].replay();b.record();b.synchronize()
                times[i].append(a.elapsed_time(b)*1000)
        empty=statistics.median(times[-1])
        for i,(name,_) in enumerate(funcs):
            row=dict(shape=load,variant=name,us=statistics.median(times[i])-empty,
                     delta_native=statistics.median([a-b for a,b in zip(times[i],times[1])]),raw_us=times[i],empty_us=empty)
            results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw_us'}),flush=True)
    del data,out,funcs,graphs,g,flush;torch.cuda.empty_cache()
Path('experiments/pv_warpgroup/results/compare-swizzle.json').write_text(json.dumps(results,indent=2)+'\n')
