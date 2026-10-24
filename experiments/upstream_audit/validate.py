"""Revalidate native dispatch against frozen upstream using complete cold graph pairs."""
import argparse,fcntl,importlib.util,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s41upstream',ROOT/'upstream.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('s41native',ROOT/'native.py')
merged=importlib.util.module_from_spec(spec);sys.modules[spec.name]=merged;spec.loader.exec_module(merged)
mods={'previous':old,'merged':merged}
cases=[]
for page in (1,64):
    for length in (1,8):
        cases.append(dict(batch=1,ctx=80000,heads='tp8',layer='full',q_len=length,page=page,dist='coherent'))
for page in (1,64):
    for batch,ctx,length,scale,dist,spread in (
        (2,9000,8,1/64,'edges','agentic'),(2,16000,8,1,'soft','agentic'),
        (2,40000,8,1,'coherent','agentic'),(1,40000,1,1,'soft',''),
        (1,120000,8,1,'coherent','')):
        c=dict(batch=batch,ctx=ctx,heads='tp8',layer='full',q_len=length,page=page,kv_scale=scale,dist=dist)
        if spread:c['spread']=spread
        cases.append(c)
for page in (1,64):
    for batch in (1,2,4,8):
        cases.append(dict(batch=batch,ctx=80000,heads='tp8',layer='sliding',q_len=1,page=page,dist='soft',spread='agentic' if batch>1 else ''))
cases += [dict(batch=1,ctx=2047,heads='dp',layer='sliding',q_len=1,page=64,dist='edges',kv_scale=1/64),
          dict(batch=1,ctx=10,heads='tp8',layer='sliding',q_len=1,page=1,dist='soft')]
# Unchanged controls outside both native dispatches.
cases += [dict(batch=3,ctx=80000,heads='tp8',layer='full',q_len=1,page=1,dist='coherent'),
          dict(batch=6,ctx=4096,heads='tp8',layer='full',q_len=8,page=64,dist='soft'),
          dict(batch=1,ctx=80000,heads='tp8',layer='sliding',q_len=8,page=1,dist='soft')]
def check(d,o,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,(label,msg)
    print('PASS',label,flush=True)
def capture(fn):
    for _ in range(3):o=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    return g,o
def compare():
    seed=41072 if sys.argv[1]=="confirm" else 41071
    flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=seed,**c);graphs={};outs={}
        cfg=d['config'];plan=old._plan(cfg,d['page_table'].shape[1],d['qkv'].device)
        span=d['page_table'].shape[1]*cfg['page_size'];span=min(span,cfg['window']) if cfg['window'] else span
        chunk=((span+plan['splits']-1)//plan['splits']+plan['block_n']-1)//plan['block_n']*plan['block_n']
        splits=plan['splits'] if plan.get('balance') else (span+chunk-1)//chunk
        native_full=not cfg['window'] and cfg['batch']*cfg['num_kv_heads']<=4 and 64<splits<=128 and not plan.get('gluon')
        for name,m in mods.items():
            check(d,m.kernel_fn(d),(i,name,'eager'))
            g,o=capture(lambda:m.kernel_fn(d));g.replay();check(d,o,(i,name,'capture'));graphs[name]=g;outs[name]=o
        raw=[]
        for r in range(80):
            order=list(mods) if r%2==0 else list(mods)[::-1]
            times={name:time(graphs[name]) for name in order};raw.append([times['previous'],times['merged'],time(eg)])
        for name in mods:graphs[name].replay();check(d,outs[name],(i,name,'after'))
        cost=statistics.median(x[2] for x in raw)
        row=dict(index=i,shape=c,native_full=native_full,native_sliding=bool(cfg['window']==2048 and cfg['q_len']==1 and cfg['batch']*cfg['num_kv_heads']<=16),splits=splits,previous_us=statistics.median(x[0] for x in raw)-cost,merged_us=statistics.median(x[1] for x in raw)-cost,
                 paired_delta_us=statistics.median(x[1]-x[0] for x in raw),raw=raw)
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
        (ROOT/('pairs-'+sys.argv[1]+'.json')).write_text(json.dumps(results,indent=2)+'\n')
        del d,graphs,outs,g,o;torch.cuda.empty_cache()
def fresh():
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=41081,**c);graphs={};outs={}
        for name,m in mods.items():graphs[name],outs[name]=capture(lambda:m.kernel_fn(d))
        for rep in range(4):
            n=reference.generate_input(seed=41082+rep,**c)
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
        (ROOT/'fresh.json').write_text(json.dumps(results,indent=2)+'\n')
        del d,n,graphs,outs;torch.cuda.empty_cache()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='confirm':
        cases=[cases[i] for i in (0,1,2,3,4,5,9,10,14,17,18,21,24,25,26)];mods=dict(reversed(list(mods.items())));mode='compare'
    if mode=='scored':
        import tools.bench as bench
        cases=[c for _,c in bench.SCORED];mode='compare'
    if mode=='suite':
        import tools.bench as bench
        bench.submission=merged
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);assert bench.test()==0
        bench.bench()
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()[mode]()
