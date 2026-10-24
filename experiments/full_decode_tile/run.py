"""Complete graph comparisons, independent reference tests, fresh-input replays."""
import argparse,fcntl,sys,json,statistics
from pathlib import Path
sys.path.insert(0,'.')
import numpy as np
import torch,triton,reference
import candidate
ROOT=Path(__file__).resolve().parent

def cases():
    result=[]
    for b,p,ctx,heads,scale,dist in [(1,1,80000,'tp8',1.,'coherent'),(2,1,80000,'tp8',1.,'coherent'),
        (16,1,80000,'tp8',1.,'coherent'),(32,1,80000,'tp8',1.,'coherent'),
        (1,64,80000,'tp8',1.,'coherent'),(2,64,9000,'tp8',1/64,'soft'),
        (1,1,10,'tp8',1.,'soft'),(1,64,4096,'dp',1/64,'coherent')]:
        result.append(dict(batch=b,page=p,ctx=ctx,heads=heads,kv_scale=scale,dist=dist,layer='full',q_len=1,spread='agentic' if b>1 else 'fixed'))
    return result

def check(data,out,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out))
    print('CHECK',label,ok,msg,flush=True);assert ok,(label,msg)

def capture(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    return g,out

def compare(args):
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda');x=torch.zeros(1,device='cuda');empty,_=capture(lambda:x.add_(1))
    a,z=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();z.record();z.synchronize();return a.elapsed_time(z)*1000
    result=[]
    for index,c in enumerate(cases()):
        if args.indices and index not in args.indices:continue
        d=reference.generate_input(seed=args.seed,**c);gs={};outs={}
        funcs={'parent':candidate.parent.kernel_fn,**{str(bn):lambda d,bn=bn:candidate.kernel(d,bn) for bn in args.widths}}
        for name,fn in funcs.items():
            out=fn(d);check(d,out,(index,name,'eager'));g,out=capture(lambda:fn(d));g.replay();check(d,out,(index,name,'captured'));gs[name]=g;outs[name]=out
        raw={str(bn):[] for bn in args.widths}
        for r in range(args.rounds):
            widths=args.widths if r%2==0 else args.widths[::-1]
            for bn in widths:
                names=('parent',str(bn));order=names if r%2==0 else names[::-1]
                t={k:time(gs[k]) for k in order};raw[str(bn)].append([t[names[0]],t[names[1]],time(empty)])
        for name in gs:gs[name].replay();check(d,outs[name],(index,name,'after'))
        for bn,values in raw.items():
            cost=statistics.median(v[2] for v in values)
            rec=dict(index=index,shape=c,width=int(bn),seed=args.seed,parent_us=statistics.median(v[0] for v in values)-cost,
                native_us=statistics.median(v[1] for v in values)-cost,paired_delta_us=statistics.median(v[1]-v[0] for v in values),raw=values)
            result.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
        del d,gs,outs,g,out;torch.cuda.empty_cache()

def fresh(args):
    result=[]
    for index,c in enumerate(cases()):
        # Bound expensive large-batch reference replays; these are already timing controls.
        if c['batch']>2:continue
        d=reference.generate_input(seed=91,**c);gs={};outs={}
        for bn in args.widths:gs[bn],outs[bn]=capture(lambda:candidate.kernel(d,bn))
        for i in range(4):
            new=reference.generate_input(seed=args.seed+i,**c)
            for k,v in d.items():
                if isinstance(v,torch.Tensor):assert v.shape==new[k].shape;v.copy_(new[k])
            if i>=2:
                if i==3:d['seq_lens'].zero_()
                else:d['seq_lens'].div_(2,rounding_mode='floor')
                pos=d['seq_lens'].long()[:,None]
                d['positions'].copy_(pos.flatten());d['cache_loc'].copy_((torch.gather(d['page_table'].long(),1,pos//c['page'])*c['page']+pos%c['page']).flatten());d['cache_seqlens'].copy_(d['seq_lens']+1)
            assert bool((d['cache_loc']>=0).all())
            for bn in args.widths:gs[bn].replay();check(d,outs[bn],('fresh',index,i,bn))
        result.append(dict(index=index,shape=c,widths=args.widths,replays_per_width=4))
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
        del d,new,gs,outs;torch.cuda.empty_cache()

def suite(args):
    import tools.bench as bench
    for bn in args.widths:
        bench.submission.kernel_fn=lambda d,bn=bn:candidate.kernel(d,bn)
        print('SUITE',bn,flush=True);assert bench.test()==0
        print('BENCH',bn,flush=True);bench.bench()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','fresh','suite']);p.add_argument('--widths',type=int,nargs='+',default=[128,256]);p.add_argument('--indices',type=int,nargs='*');p.add_argument('--rounds',type=int,default=60);p.add_argument('--seed',type=int,default=28103);p.add_argument('--name',default='initial');args=p.parse_args()
    for bn in args.widths:candidate.extension(bn)
    if args.mode=='suite':suite(args)
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()[args.mode](args)
