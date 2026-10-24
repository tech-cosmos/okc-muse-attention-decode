"""Offline diagnosis of small-full merge PDL; all timed paths compute full outputs."""
import argparse
import fcntl
import importlib.util
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, '.')
import numpy as np
import torch
import triton
import reference
ROOT=Path(__file__).resolve().parent


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name, ROOT/filename)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

main=load('s27_main','main_snapshot.py')
native=load('s27_native','production_snapshot.py')
serial_main=load('s27_serial_main','main_snapshot.py')
serial_native=load('s27_serial_native','production_snapshot.py')

class SerialNative:
    def combine_full(self,*args):
        # Only the merge's launch attribute/wait specialization changes.
        return native._NATIVE_MERGE.combine_full(*args[:-1],False)

class SerialTriton:
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            kwargs['PDL']=False;kwargs['launch_pdl']=False
            return main._combine[grid](*args,**kwargs)
        return run

serial_native._NATIVE_MERGE=SerialNative()
serial_main._combine=SerialTriton()
MODULES={'main':main,'native':native,'serial_main':serial_main,'serial_native':serial_native}
PAIRS=[('main','native'),('native','serial_native'),('main','serial_main'),('serial_main','serial_native')]

def cases():
    result=[]
    for page in (1,64):
        for b,L,ctx,spread in [(1,1,80000,'fixed'),(1,8,80000,'fixed'),
                              (2,8,8192,'agentic'),(2,8,9000,'agentic'),
                              (2,8,10000,'agentic'),(2,8,16000,'agentic'),
                              (2,8,40000,'agentic'),(2,8,80000,'agentic'),
                              (2,8,9000,'fixed'),(1,8,9000,'fixed')]:
            result.append(dict(batch=b,q_len=L,ctx=ctx,heads='tp8',layer='full',
                               page=page,spread=spread,dist='coherent'))
    return result

def path(data):
    c=data['config']; span=data['page_table'].shape[1]*c['page_size']
    p=native._plan(c,data['page_table'].shape[1],data['qkv'].device)
    chunk=triton.cdiv(triton.cdiv(span,p['splits']),p['block_n'])*p['block_n']
    s=triton.cdiv(span,chunk)
    assert c['batch']*c['num_kv_heads']<=4 and 64<s<=128,(c,s)
    return dict(splits=s,chunk=chunk,span=span,merge_ctas=c['batch']*c['q_len']*c['num_q_heads'])

def capture(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    return g,out

def check(data,out,where):
    torch.cuda.synchronize()
    ok,msg=reference.check_implementation(data,reference.collect(data,out))
    assert ok,(where,msg)

def interval(raw):
    d=np.array([r[1]-r[0] for r in raw]).reshape(-1,2).mean(axis=1)
    rng=np.random.default_rng(27103)
    samples=np.median(d[rng.integers(len(d),size=(2000,len(d)))],axis=1)
    return float(np.median(d)),np.quantile(samples,[.025,.975]).tolist()

def compare(args):
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
    x=torch.zeros(1,device='cuda');empty,_=capture(lambda:x.add_(1))
    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    def time(g):
        flush.zero_();a.record();g.replay();z.record();z.synchronize()
        return a.elapsed_time(z)*1000
    results=[]
    for index,c in enumerate(cases()):
        if args.indices and index not in args.indices:continue
        data=reference.generate_input(seed=args.seed,**c);meta=path(data)
        graphs={};outputs={}
        for name,m in MODULES.items():
            out=m.kernel_fn(data);check(data,out,(index,name,'eager'))
            g,out=capture(lambda:m.kernel_fn(data));g.replay();check(data,out,(index,name,'capture'))
            graphs[name]=g;outputs[name]=out
        entries={pair:[] for pair in PAIRS}
        for r in range(args.rounds):
            # Rotate which contrast goes first; every contrast has adjacent AB/BA replays.
            contrasts=PAIRS[r%4:]+PAIRS[:r%4]
            for pair in contrasts:
                order=pair if r%2==0 else pair[::-1]
                t={name:time(graphs[name]) for name in order}
                entries[pair].append([t[pair[0]],t[pair[1]],time(empty)])
        for name,g in graphs.items():g.replay();check(data,outputs[name],(index,name,'after timing'))
        contrasts=[]
        for pair,raw in entries.items():
            cost=statistics.median(r[2] for r in raw);delta,ci=interval(raw)
            contrasts.append(dict(pair=pair,left_us=statistics.median(r[0] for r in raw)-cost,
                                  right_us=statistics.median(r[1] for r in raw)-cost,
                                  paired_delta_us=statistics.median(r[1]-r[0] for r in raw),
                                  balanced_block_delta_us=delta,block_bootstrap_ci=ci,raw=raw))
        rec=dict(index=index,shape=c,seed=args.seed,**meta,contrasts=contrasts)
        results.append(rec)
        print(json.dumps({**rec,'contrasts':[{k:v for k,v in row.items() if k!='raw'} for row in contrasts]}),flush=True)
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(results,indent=2)+'\n')
        del data,graphs,outputs,g,out;torch.cuda.empty_cache()

def fresh(args):
    result=[]
    for index,c in enumerate(cases()):
        if args.indices and index not in args.indices:continue
        data=reference.generate_input(seed=101,**c);meta=path(data)
        graphs={};outputs={}
        for name,m in MODULES.items():graphs[name],outputs[name]=capture(lambda:m.kernel_fn(data))
        for i in range(4):
            new=reference.generate_input(seed=800+i,**c)
            for k,v in data.items():
                if isinstance(v,torch.Tensor):
                    assert v.shape==new[k].shape
                    v.copy_(new[k])
            if i>=2:
                gen=torch.Generator(device='cuda').manual_seed(27000+index)
                data['seq_lens'].copy_((torch.rand((c['batch'],),generator=gen,device='cuda')*(new['seq_lens']+1)).to(torch.int32))
                if i==3:data['seq_lens'].zero_()
                pos=data['seq_lens'].long()[:,None]+torch.arange(c['q_len'],device='cuda')
                data['positions'].copy_(pos.flatten())
                data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//c['page'])*c['page']+pos%c['page']).flatten())
                data['cache_seqlens'].copy_(data['seq_lens']+c['q_len'])
            assert bool((data['cache_loc']>=0).all())
            for name,g in graphs.items():g.replay();check(data,outputs[name],(index,i,name));print('PASS replay',index,i,name,flush=True)
        result.append(dict(index=index,shape=c,**meta,replays_per_implementation=4))
        (ROOT/'results'/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
        del data,new,graphs,outputs,g;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','fresh']);p.add_argument('--seed',type=int,default=27103)
    p.add_argument('--rounds',type=int,default=80);p.add_argument('--name',default='initial');p.add_argument('--indices',type=int,nargs='*')
    args=p.parse_args();assert args.rounds%2==0
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (compare if args.mode=='compare' else fresh)(args)
