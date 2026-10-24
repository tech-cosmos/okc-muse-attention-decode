"""Validate main gfuse integration with native kernels using complete cold graph pairs."""
import argparse,fcntl,importlib.util,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s42previous',ROOT/'previous.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('s42merged',ROOT/'merged.py')
merged=importlib.util.module_from_spec(spec);sys.modules[spec.name]=merged;spec.loader.exec_module(merged)
mods={'previous':old,'merged':merged}
cases=[]
for batch,ctx,page,heads,scale,dist,spread in (
    (16,80000,1,'tp8',1,'coherent','agentic'),
    (32,80000,1,'tp8',1,'coherent','agentic'),
    (6,30000,1,'tp8',1,'soft','mixed'),
    (8,30000,1,'tp8',1,'coherent',''),
    (12,80000,1,'tp8',1,'coherent','agentic'),
    (20,6000,64,'tp8',1/64,'edges',''),
    (24,9000,64,'tp8',1,'normal','mixed'),
    (48,16000,64,'tp8',1,'soft','agentic'),
    (64,40000,1,'tp8',1,'coherent','mixed'),
    (16,20000,64,'tp8',1/64,'edges',''),
    (2,12000,64,'dp',1/64,'edges',''),
    (4,20000,1,'dp',1,'soft','mixed'),
    (1,80000,1,'tp8',1,'coherent',''),
):
    c=dict(batch=batch,ctx=ctx,heads=heads,layer='full',q_len=8,page=page,kv_scale=scale,dist=dist)
    if spread:c['spread']=spread
    cases.append(c)
cases += [dict(batch=1,ctx=80000,heads='tp8',layer='sliding',q_len=1,page=1,dist='soft'),
          dict(batch=1,ctx=80000,heads='tp8',layer='full',q_len=1,page=1,dist='coherent')]
def check(d,o,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,(label,msg)
    print('PASS',label,flush=True)
def capture(fn):
    for _ in range(3):o=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    return g,o
def compare():
    seed=42072 if sys.argv[1]=="confirm" else 42071
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
        row=dict(index=i,shape=c,native_full=native_full,native_sliding=bool(cfg['window']==2048 and cfg['q_len']==1 and cfg['batch']*cfg['num_kv_heads']<=16),splits=splits,gfuse=bool(plan.get('gluon') and plan['splits']*cfg['batch']*(cfg['num_kv_heads']//2)<=2*merged._num_sms(d['qkv'].device)),previous_us=statistics.median(x[0] for x in raw)-cost,merged_us=statistics.median(x[1] for x in raw)-cost,
                 paired_delta_us=statistics.median(x[1]-x[0] for x in raw),raw=raw)
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
        (ROOT/('pairs-'+sys.argv[1]+'.json')).write_text(json.dumps(results,indent=2)+'\n')
        del d,graphs,outs,g,o;torch.cuda.empty_cache()
def fresh():
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=42081,**c);graphs={};outs={}
        for name,m in mods.items():graphs[name],outs[name]=capture(lambda:m.kernel_fn(d))
        for rep in range(4):
            n=reference.generate_input(seed=42082+rep,**c)
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
        cases=[cases[i] for i in (0,1,2,3,4,5,9,10,11,12,13,14)];mods=dict(reversed(list(mods.items())));mode='compare'
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
