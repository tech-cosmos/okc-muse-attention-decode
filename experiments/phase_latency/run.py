"""Complete output controls plus offline per-warp phase cycle measurements."""
import argparse,fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference
from experiments.phase_latency import candidate as c
ROOT=Path(__file__).resolve().parent

def cases():
    rows=[]
    for b,page in ((1,1),(8,1),(1,64),(8,64)):
        rows.append(dict(batch=b,ctx=80000,heads='tp8',layer='sliding',q_len=1,page=page,dist='soft',spread='agentic' if b>1 else 'fixed'))
    rows += [dict(batch=1,ctx=4096,heads='dp',layer='sliding',q_len=1,page=64,kv_scale=1/64,dist='coherent'),dict(batch=1,ctx=10,heads='tp8',layer='sliding',q_len=1,page=1,dist='edges'),dict(batch=3,ctx=2047,heads='tp8',layer='sliding',q_len=1,page=1,dist='edges')]
    return rows

def check(d,out,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(label,msg)
    print('PASS',label,flush=True)

def capture(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    return g,out

def measure(args):
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda');x=torch.zeros(1,device='cuda');empty,_=capture(lambda:x.add_(1))
    a,z=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();z.record();z.synchronize();return a.elapsed_time(z)*1000
    results=[]
    for index,shape in enumerate(cases()):
        if args.indices is not None and index not in args.indices:continue
        d=reference.generate_input(seed=args.seed,**shape);assert c.supported(d['config'])
        gs={};outs={};traces={}
        modes=['parent','same_code']+list(range(-1,12))
        if args.reverse:modes.reverse()
        for mode in modes:
            fn=(lambda:c.parent.kernel_fn(d)) if isinstance(mode,str) else (lambda mode=mode:c.run(d,mode))
            ret=fn();out=ret if isinstance(mode,str) else ret[0];check(d,out,(index,mode,'eager'))
            g,ret=capture(fn);g.replay();out=ret if isinstance(mode,str) else ret[0];check(d,out,(index,mode,'capture'));gs[mode]=g;outs[mode]=out
            if isinstance(mode,int) and mode>=0:traces[mode]=ret[1]
        raw={mode:[] for mode in modes if mode!='parent'}
        for r in range(args.rounds):
            order=list(raw)
            if r%2:order.reverse()
            for mode in order:
                names=['parent',mode] if r%2==0 else [mode,'parent'];t={k:time(gs[k]) for k in names}
                raw[mode].append([t['parent'],t[mode],time(empty)])
        # Separate cold-replay samples: traces are instrumentation, not graph times.
        cycles={str(mode):[] for mode in traces}
        for r in range(args.samples):
            order=list(traces)
            if r%2:order.reverse()
            for mode in order:
                flush.zero_();gs[mode].replay();torch.cuda.synchronize()
                trace=traces[mode].cpu();delta=trace[...,1]-trace[...,0];assert bool((delta>=0).all()),(mode,delta.min())
                cycles[str(mode)].append(delta.tolist())
        for mode in gs:gs[mode].replay();check(d,outs[mode],(index,mode,'after'))
        timing={}
        for mode,values in raw.items():
            cost=statistics.median(v[2] for v in values)
            timing[str(mode)]=dict(parent_us=statistics.median(v[0] for v in values)-cost,candidate_us=statistics.median(v[1] for v in values)-cost,paired_delta_us=statistics.median(v[1]-v[0] for v in values),raw=values)
        rec=dict(index=index,shape=shape,seq_lens=d['seq_lens'].cpu().tolist(),seed=args.seed,reverse_capture=args.reverse,timing=timing,cycles=cycles)
        results.append(rec);(ROOT/'results'/f'{args.name}.json').write_text(json.dumps(results)+'\n')
        print('TIMING',index,json.dumps({m:{k:v for k,v in t.items() if k!='raw'} for m,t in timing.items()}),flush=True)
        del d,gs,outs,traces,trace,delta,g,out,ret;torch.cuda.empty_cache()

def fresh(args):
    results=[]
    for index,shape in enumerate(cases()):
        d=reference.generate_input(seed=91,**shape)
        graphs=[capture(lambda mode=mode:c.run(d,mode)) for mode in range(-1,12)]
        for i in range(4):
            new=reference.generate_input(seed=args.seed+i,**shape)
            for k,v in d.items():
                if isinstance(v,torch.Tensor):assert v.shape==new[k].shape;v.copy_(new[k])
            if i>=2:
                if i==3:d['seq_lens'].zero_()
                else:d['seq_lens'].div_(2,rounding_mode='floor')
                pos=d['seq_lens'].long()[:,None]
                d['positions'].copy_(pos.flatten());d['cache_loc'].copy_((torch.gather(d['page_table'].long(),1,pos//shape['page'])*shape['page']+pos%shape['page']).flatten());d['cache_seqlens'].copy_(d['seq_lens']+1)
            assert bool((d['cache_loc']>=0).all())
            for mode,(g,(out,trace)) in zip(range(-1,12),graphs):
                g.replay();check(d,out,('fresh',index,i,mode))
        results.append(dict(index=index,shape=shape,replays=52));(ROOT/'results'/f'{args.name}.json').write_text(json.dumps(results,indent=2)+'\n')
        del d,new,graphs,g,out,trace;torch.cuda.empty_cache()

def suite(args):
    import tools.bench as bench
    bench.submission=c
    assert bench.test()==0
    bench.bench()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['measure','fresh','suite']);p.add_argument('--indices',type=int,nargs='*');p.add_argument('--rounds',type=int,default=40);p.add_argument('--samples',type=int,default=12);p.add_argument('--seed',type=int,default=32001);p.add_argument('--name',default='initial');p.add_argument('--reverse',action='store_true');args=p.parse_args();assert args.rounds%2==0
    if args.mode=='suite':suite(args)
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()[args.mode](args)
