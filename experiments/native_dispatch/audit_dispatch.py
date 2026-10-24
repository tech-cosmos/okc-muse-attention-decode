"""Offline complete-kernel validation and paired graph generalization audit."""
import argparse
import fcntl
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '.')
import numpy as np
import torch
import triton
import submission
import reference
from experiments.native_dispatch import main_snapshot as main

ROOT = Path(__file__).resolve().parent

def cases():
    result = []
    def add(b, ctx, layer='sliding', length=1, page=1, heads='tp8', scale=1., spread=None):
        result.append(dict(batch=b, ctx=ctx, heads=heads, layer=layer, q_len=length,
                           page=page, kv_scale=scale, dist='soft' if layer=='sliding' else 'coherent',
                           spread=spread or ('fixed' if b==1 else 'agentic')))
    for page in (1,64):
        for b in (1,7,8,9): add(b,80000,page=page)
    for i,ctx in enumerate((63,64,65,2047,2048,2049)):
        for b in (1,8): add(b,ctx,page=1 if i%2==0 else 64,spread='fixed')
    for page in (1,64):
        for b in (1,2): add(b,4000,page=page,heads='dp',scale=1/64)
    for ctx in (40000,120000):
        for b in (2,6): add(b,ctx,page=64,scale=1/64,spread='mixed')
    for length in (1,8):
        for b in (1,2,3): add(b,80000,layer='full',length=length)
        for ctx in (40000,120000):
            for b in (1,2): add(b,ctx,layer='full',length=length,page=64,scale=1/64)
        for ctx in (1024,9000):
            for b in (1,2): add(b,ctx,layer='full',length=length,page=64)
    for length in (1,8):
        for b in ((1,) if length==1 else (1,2)):
            for page in (1,64):
                for ctx in (8191,8192):
                    add(b,ctx,layer="full",length=length,page=page,spread="fixed")
    for ctx in (8184,8185):
        add(1,ctx,layer="full",length=8,page=1,spread="fixed")
    return result

def path(data):
    c=data['config']; b,h=c['batch'],c['num_kv_heads']; mp=data['page_table'].shape[1]
    if c['window'] and c['q_len']==1 and b*h<=16:
        return 'native_sliding',min(32,triton.cdiv(mp*c['page_size'],64))
    p=submission._plan(c,mp,data['qkv'].device)
    if p.get('gluon'): return 'gluon_fallback',p['splits']+1
    span=min(mp*c['page_size'],c['window']) if c['window'] else mp*c['page_size']
    chunk=triton.cdiv(triton.cdiv(span,p['splits']),p['block_n'])*p['block_n']
    splits=p['splits'] if p.get('balance') else triton.cdiv(span,chunk)
    return ('native_full_merge' if not c['window'] and b*h<=4 and 64<splits<=128 else 'triton_fallback'),splits

def capture(fn):
    for _ in range(3): out=fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=fn()
    return graph,out

def check(data,out,context):
    torch.cuda.synchronize()
    ok,msg=reference.check_implementation(data,reference.collect(data,out))
    assert ok,(context,msg)

def interval(raw):
    # Resample adjacent AB/BA blocks, retaining order-balancing within blocks.
    d=np.array([r[1]-r[0] for r in raw]).reshape(-1,2).mean(axis=1)
    rng=np.random.default_rng(25103)
    med=np.median(d[rng.integers(0,len(d),size=(2000,len(d)))],axis=1)
    return float(np.median(d)),np.quantile(med,[.025,.975]).tolist()

def compare(args):
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
    x=torch.zeros(1,device='cuda'); empty,_=capture(lambda:x.add_(1))
    a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
    def time(graph):
        flush.zero_(); a.record(); graph.replay(); z.record(); z.synchronize()
        return a.elapsed_time(z)*1000
    results=[]
    for index,c in enumerate(cases()):
        if args.indices and index not in args.indices: continue
        data=reference.generate_input(seed=args.seed,**c)
        graphs=[]; outputs=[]
        for module in (main,submission):
            out=module.kernel_fn(data); check(data,out,(index,module.__name__,'eager'))
            graph,out=capture(lambda:module.kernel_fn(data))
            graph.replay();check(data,out,(index,module.__name__,'graph'))
            graphs.append(graph);outputs.append(out)
        raw=[]
        for pair in range(args.rounds):
            order=(0,1) if pair%2==0 else (1,0)
            times={j:time(graphs[j]) for j in order}
            raw.append([times[0],times[1],time(empty)])
        for graph,out in zip(graphs,outputs):
            graph.replay();check(data,out,(index,'after timing'))
        cost=statistics.median(r[2] for r in raw)
        delta,ci=interval(raw); dispatch,splits=path(data)
        rec=dict(index=index,shape=c,dispatch=dispatch,splits=splits,
                 main_us=statistics.median(r[0] for r in raw)-cost,
                 codex_us=statistics.median(r[1] for r in raw)-cost,
                 paired_delta_us=statistics.median(r[1]-r[0] for r in raw),
                 balanced_block_delta_us=delta,block_bootstrap_ci=ci,raw=raw)
        results.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(results,indent=2)+'\n')
        del graphs,outputs,graph,out,data
        torch.cuda.empty_cache()

def fresh(args):
    # Both sides replay on changing tensors; not a performance measurement.
    chosen=(0,2,3,8,12,18,20,22,28,29,34,40,41,44,46,48,50,51,54,55,58,59,62,63)
    records=[]
    for index in chosen:
        if args.indices and index not in args.indices: continue
        c=cases()[index];data=reference.generate_input(seed=101,**c)
        graphs=[];outputs=[]
        for module in (main,submission):
            graph,out=capture(lambda:module.kernel_fn(data));graphs.append(graph);outputs.append(out)
        for i in range(4):
            new=reference.generate_input(seed=300+i,**c)
            for key,val in data.items():
                if isinstance(val,torch.Tensor):
                    assert val.shape==new[key].shape,(key,val.shape,new[key].shape)
                    val.copy_(new[key])
            if i>=2:
                # Shorten each generated request; never extend into its unassigned pages.
                gen=torch.Generator(device='cuda').manual_seed(25000+index)
                data['seq_lens'].copy_((torch.rand((c['batch'],),generator=gen,device='cuda')*(new['seq_lens']+1)).to(torch.int32))
                if i==3:data['seq_lens'].zero_()
                pos=data['seq_lens'].long()[:,None]+torch.arange(c['q_len'],device='cuda')
                data['positions'].copy_(pos.flatten())
                data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//c['page'])*c['page']+pos%c['page']).flatten())
                data['cache_seqlens'].copy_(data['seq_lens']+c['q_len'])
            assert bool((data['cache_loc']>=0).all()),(index,i,'invalid new row')
            for j,(graph,out) in enumerate(zip(graphs,outputs)):
                print('replay',index,i,j,flush=True)
                try:
                    graph.replay();check(data,out,(index,i,j))
                except Exception as exc:
                    raise RuntimeError(('fresh replay failed',index,i,j)) from exc
        dispatch,splits=path(data)
        rec=dict(index=index,shape=c,dispatch=dispatch,splits=splits,replays_per_implementation=4)
        records.append(rec);print('PASS fresh '+json.dumps(rec),flush=True)
        (ROOT/'results'/'fresh.json').write_text(json.dumps(records,indent=2)+'\n')
        del data,new,graphs,outputs,graph,out;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','fresh']);p.add_argument('--seed',type=int,default=25103)
    p.add_argument('--rounds',type=int,default=80);p.add_argument('--name',default='initial');p.add_argument('--indices',type=int,nargs='*')
    args=p.parse_args();assert args.rounds%2==0
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (compare if args.mode=='compare' else fresh)(args)
