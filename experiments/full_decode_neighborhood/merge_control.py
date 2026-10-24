"""Complete-graph control: change only which existing merge consumes native partials."""
import argparse
import fcntl
import json
import statistics
from pathlib import Path
import sys
sys.path.insert(0, '.')
import torch
import triton
import reference
import candidate
import native_merge_control
from run import capture, check

ROOT = Path(__file__).resolve().parent

def cases():
    return [dict(batch=b,ctx=c,page=p,heads='tp8',kv_scale=1.,dist='coherent',
                 layer='full',q_len=1,spread='fixed' if b==1 else 'agentic')
            for b in (1,2) for c in (4096,8192,9000,16000) for p in (1,64)]

def compare():
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
    x=torch.zeros(1,device='cuda');empty,_=capture(lambda:x.add_(1))
    a,z=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();z.record();z.synchronize()
        return a.elapsed_time(z)*1000
    result=[]
    for index,c in enumerate(cases()):
        d=reference.generate_input(seed=29401,**c)
        funcs={'parent':candidate.parent.kernel_fn,'tile':lambda d:candidate.kernel(d,128),
               'composed':native_merge_control.kernel}
        gs={};outs={}
        for name,fn in funcs.items():
            check(d,fn(d),(index,name,'eager'))
            gs[name],outs[name]=capture(lambda:fn(d))
            gs[name].replay();check(d,outs[name],(index,name,'captured'))
        span=d['page_table'].shape[1]*c['page'];s=triton.cdiv(span,128)
        raw={k:[] for k in ('parent','tile')}
        for r in range(80):
            for base in (('parent','tile') if r%2==0 else ('tile','parent')):
                order=(base,'composed') if r%2==0 else ('composed',base)
                t={name:time(gs[name]) for name in order}
                raw[base].append([t[base],t['composed'],time(empty)])
        for name in gs:
            gs[name].replay();check(d,outs[name],(index,name,'after'))
        for base,values in raw.items():
            cost=statistics.median(v[2] for v in values)
            rec=dict(index=index,shape=c,span=span,splits=s,uses_native_merge=c['batch']*2<=4 and 64<s<=128,
                     baseline=base,baseline_us=statistics.median(v[0] for v in values)-cost,
                     composed_us=statistics.median(v[1] for v in values)-cost,
                     paired_delta_us=statistics.median(v[1]-v[0] for v in values),raw=values)
            result.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
        (ROOT/'results/merge-control.json').write_text(json.dumps(result,indent=2)+'\n')
        del d,gs,outs;torch.cuda.empty_cache()

def fresh():
    total=0
    for index,c in enumerate(cases()):
        d=reference.generate_input(seed=29401,**c)
        g,out=capture(lambda:native_merge_control.kernel(d))
        for i in range(4):
            new=reference.generate_input(seed=29501+i,**c)
            for k,v in d.items():
                if isinstance(v,torch.Tensor):
                    assert v.shape==new[k].shape
                    v.copy_(new[k])
            if i>=2:
                if i==3:d['seq_lens'].zero_()
                else:d['seq_lens'].div_(2,rounding_mode='floor')
                pos=d['seq_lens'].long()[:,None]
                d['positions'].copy_(pos.flatten())
                d['cache_loc'].copy_((torch.gather(d['page_table'].long(),1,pos//c['page'])*c['page']+pos%c['page']).flatten())
                d['cache_seqlens'].copy_(d['seq_lens']+1)
            assert bool((d['cache_loc']>=0).all())
            g.replay();check(d,out,('composed-fresh',index,i));total+=1
        del d,new,g,out;torch.cuda.empty_cache()
    print('FRESH_TOTAL',total,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','fresh','suite']);args=p.parse_args()
    candidate.extension(128)
    if args.mode=='suite':
        import tools.bench as bench
        bench.submission.kernel_fn=native_merge_control.kernel
        assert bench.test()==0
        bench.bench()
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            globals()[args.mode]()
