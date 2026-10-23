import sys,json,itertools
sys.path.insert(0,'.')
from tools import bench
import reference,torch
from experiments.gather import gather
for batch in (16,32):
    c=dict(batch=batch,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent',spread='agentic')
    data=reference.generate_input(seed=99,**c)
    expected=None;results=[]
    for heads,prod,st,ctas in itertools.product((2,),(2,4),(2,3,4),(528,1056,1584)):
        s=max(1,ctas//(batch*2//heads))
        opts=dict(splits=s,heads=heads,stages=st,producer=prod)
        val=gather.run(data,**opts).long().sum().item()%(2**32)
        if expected is None:expected=val
        assert val==expected,(opts,val,expected)
        t=bench.time_graph(lambda:gather.run(data,**opts))-bench.empty_cost()
        results.append(dict(us=t,**opts))
        print(batch,round(t,3),opts,flush=True)
    results.sort(key=lambda r:r['us'])
    print('BEST',batch,results[:8],flush=True)
    open(f'experiments/gather/results/gather-multi-{batch}.json','w').write(json.dumps(results,indent=2))
    del data;torch.cuda.empty_cache()
