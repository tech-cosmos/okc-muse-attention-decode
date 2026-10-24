"""Compare complete native decode graphs with frozen production or natural-V control."""
import argparse,fcntl,importlib.util,json,statistics,sys,os
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
ROOT=Path(__file__).resolve().parent
from experiments.deferred_denominator import native
CONTROL=os.environ.get('CONTROL','production')
if CONTROL=='natural':
    saved=os.environ.get('TDEC_VARIANT');os.environ['TDEC_VARIANT']='natural'
    spec=importlib.util.spec_from_file_location('s36natural',ROOT/'native.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    if saved is None:del os.environ['TDEC_VARIANT']
    else:os.environ['TDEC_VARIANT']=saved
else:control=submission
mods={'previous':control,'merged':native}
cases=[dict(batch=b,ctx=80000,heads='tp8',layer='full',q_len=1,page=1,spread='fixed' if b==1 else 'agentic',dist='coherent') for b in (1,8,16,32)]
cases += [dict(batch=1,ctx=80000,heads='tp8',layer='full',q_len=1,page=64,dist='coherent'),
          dict(batch=4,ctx=6000,heads='tp8',layer='full',q_len=1,page=64,kv_scale=1/64,dist='edges'),
          dict(batch=2,ctx=9000,heads='tp8',layer='full',q_len=1,page=64,spread='mixed',dist='soft'),
          dict(batch=1,ctx=4096,heads='dp',layer='full',q_len=1,page=64,kv_scale=1/64,dist='coherent')]
def check(d,o,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,(label,msg)
    print('PASS',label,flush=True)
def capture(fn):
    for _ in range(3):o=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    return g,o
def compare():
    flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=36171,**c);graphs={};outs={}
        for name,m in mods.items():
            check(d,m.kernel_fn(d),(i,name,'eager'))
            g,o=capture(lambda:m.kernel_fn(d));g.replay();check(d,o,(i,name,'capture'));graphs[name]=g;outs[name]=o
        raw=[]
        for r in range(60):
            order=list(mods) if r%2==0 else list(mods)[::-1]
            times={name:time(graphs[name]) for name in order};raw.append([times['previous'],times['merged'],time(eg)])
        for name in mods:graphs[name].replay();check(d,outs[name],(i,name,'after'))
        cost=statistics.median(x[2] for x in raw)
        row=dict(index=i,shape=c,previous_us=statistics.median(x[0] for x in raw)-cost,merged_us=statistics.median(x[1] for x in raw)-cost,
                 paired_delta_us=statistics.median(x[1]-x[0] for x in raw),raw=raw)
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
        (ROOT/(native.VARIANT+'-'+CONTROL+'-pairs.json')).write_text(json.dumps(results,indent=2)+'\n')
        del d,graphs,outs,g,o;torch.cuda.empty_cache()
def fresh():
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=36181,**c);graphs={};outs={}
        for name,m in mods.items():graphs[name],outs[name]=capture(lambda:m.kernel_fn(d))
        for rep in range(4):
            n=reference.generate_input(seed=36182+rep,**c)
            for key,v in d.items():
                if isinstance(v,torch.Tensor):v.copy_(n[key])
            if rep>=2:
                if rep==3:d['seq_lens'].zero_()
                else:d['seq_lens'].div_(2,rounding_mode='floor')
                p=d['seq_lens'].long()[:,None]+torch.arange(c['q_len'],device='cuda')[None,:]
                d['positions'].copy_(p.flatten());d['cache_seqlens'].copy_(d['seq_lens']+c['q_len'])
                d['cache_loc'].copy_((torch.gather(d['page_table'].long(),1,p//c['page'])*c['page']+p%c['page']).flatten())
            assert (d['cache_loc']>=0).all()
            for name in mods:graphs[name].replay();check(d,outs[name],('fresh',i,name,rep))
        results.append(dict(index=i,shape=c,replays_per_source=4))
        (ROOT/(native.VARIANT+'-fresh.json')).write_text(json.dumps(results,indent=2)+'\n')
        del d,n,graphs,outs;torch.cuda.empty_cache()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='suite':
        import tools.bench as bench
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);bench.submission=native;assert bench.test()==0
        bench.bench()
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()[mode]()
