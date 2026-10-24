"""Native CTA traversal experiment. All compared graphs compute complete outputs."""
import argparse,fcntl,importlib.util,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference
ROOT=Path(__file__).resolve().parent

def load(name,file):
    sp=importlib.util.spec_from_file_location(name,ROOT/file)
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
parent=load('s30parent','production_snapshot.py')
neighbor=load('s30neighbor','neighbor.py')
control=load('s30control','production_snapshot.py')
MODULES={'parent':parent,'neighbor':neighbor,'control':control}

def cases():
    rows=[]
    for page in (1,64):
        for b in (1,2,4,8):
            rows.append(dict(batch=b,ctx=80000,heads='tp8',layer='sliding',q_len=1,page=page,dist='soft',spread='agentic' if b>1 else 'fixed'))
    for b,ctx,page,heads,scale,dist in [(1,10,1,'tp8',1,'edges'),(2,63,64,'tp8',1/64,'soft'),(3,2047,1,'tp8',1,'edges'),(8,2049,64,'tp8',1/64,'edges'),(1,4096,1,'dp',1,'soft'),(1,4096,64,'dp',1/64,'coherent'),(2,120000,1,'tp8',1/64,'coherent'),(4,4000,64,'tp8',1,'soft')]:
        rows.append(dict(batch=b,ctx=ctx,heads=heads,layer='sliding',q_len=1,page=page,kv_scale=scale,dist=dist,spread='mixed' if b>1 else 'fixed'))
    return rows

def check(d,out,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(label,msg)
    print('PASS',label,flush=True)

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
        if args.indices is not None and index not in args.indices:continue
        d=reference.generate_input(seed=args.seed,**c);gs={};outs={}
        order=list(MODULES)
        if args.reverse:order.reverse()
        for name in order:
            m=MODULES[name];check(d,m.kernel_fn(d),(index,name,'eager'))
            g,out=capture(lambda:m.kernel_fn(d));g.replay();check(d,out,(index,name,'capture'));gs[name]=g;outs[name]=out
        raw={name:[] for name in ('neighbor','control')}
        for r in range(args.rounds):
            for name in (['neighbor','control'] if r%2==0 else ['control','neighbor']):
                names=['parent',name];order=names if r%2==0 else names[::-1]
                t={k:time(gs[k]) for k in order};raw[name].append([t['parent'],t[name],time(empty)])
        for name in gs:gs[name].replay();check(d,outs[name],(index,name,'after'))
        for name,values in raw.items():
            cost=statistics.median(v[2] for v in values)
            rec=dict(index=index,shape=c,variant=name,seed=args.seed,reverse_capture=args.reverse,parent_us=statistics.median(v[0] for v in values)-cost,
                candidate_us=statistics.median(v[1] for v in values)-cost,paired_delta_us=statistics.median(v[1]-v[0] for v in values),raw=values)
            result.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
        del d,gs,outs,g,out;torch.cuda.empty_cache()

def fresh(args):
    result=[]
    for index,c in enumerate(cases()):
        if args.indices is not None and index not in args.indices:continue
        d=reference.generate_input(seed=91,**c);g,out=capture(lambda:neighbor.kernel_fn(d))
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
            g.replay();check(d,out,('fresh',index,i))
        result.append(dict(index=index,shape=c,replays=4))
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
        del d,new,g,out;torch.cuda.empty_cache()

def suite(args):
    import tools.bench as bench
    bench.submission=neighbor
    assert bench.test()==0
    bench.bench()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','fresh','suite']);p.add_argument('--indices',type=int,nargs='*');p.add_argument('--rounds',type=int,default=80);p.add_argument('--seed',type=int,default=30103);p.add_argument('--name',default='initial');p.add_argument('--reverse',action='store_true');args=p.parse_args();assert args.rounds%2==0
    if args.mode=='suite':suite(args)
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()[args.mode](args)
