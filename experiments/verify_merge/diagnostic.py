"""Offline only: decomposition of the unchanged full-verify launch chain.
Omission mode returns partials, never a valid kernel_fn output or submission.
"""
import argparse,fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission as sub,reference
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=80);p.add_argument('--seed',type=int,default=103);a=p.parse_args()
original=sub._combine
class Merge:
    def __init__(self,mode):self.mode=mode;self.args=None
    def __getitem__(self,grid):
        def call(*args,**kw):
            self.args=args
            if self.mode=='omit':return
            if self.mode=='serial':kw['PDL']=False;kw['launch_pdl']=False
            return original[grid](*args,**kw)
        return call

def run(data,m):
    sub._combine=m
    try:return sub.kernel_fn(data)
    finally:sub._combine=original

def validate(data,out,m):
    if m.mode=='omit':
        qkv,pa,pm,vs,_=m.args
        ma=pm[...,0].amax(2,keepdim=True);w=(pm[...,0]-ma).exp2()
        den=(w*pm[...,1]).sum(2)
        v=(pa*w[...,None]).sum(2)/den[...,None]*vs
        cfg=data['config'];b=cfg['batch'];l=cfg['q_len'];hq=cfg['num_q_heads'];hkv=cfg['num_kv_heads'];g=hq//hkv
        v=v.reshape(b,hkv,l,g,64).permute(0,2,1,3,4).reshape(b,l,hq,64)
        v=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)
        gate=qkv[...,(hq+2*hkv)*64:].reshape(b,l,hq,64).float()
        out=(v*gate.sigmoid()).to(torch.bfloat16).reshape(b,l,hq*64)
    ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,msg
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
 def capture(f):
    for _ in range(3):out=f()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=f()
    g.replay();torch.cuda.synchronize();return g,out
 x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
 def time(g):
    flush.zero_();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record();g.replay();e.record();e.synchronize();return s.elapsed_time(e)*1000
 for b in (16,17,24,32,48,64):
  c=dict(batch=b,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent',spread='agentic')
  data=reference.generate_input(seed=a.seed,**c)
  for mode in ('omit','serial'):
   merges=[Merge('full'),Merge(mode)];gs=[];outs=[]
   for m in merges:
    out=run(data,m);torch.cuda.synchronize();validate(data,out,m)
    g,out=capture(lambda:run(data,m));validate(data,out,m);gs.append(g);outs.append(out)
   raw=[]
   for r in range(a.rounds):
    vals={j:time(gs[j]) for j in ([0,1] if r%2==0 else [1,0])};raw.append([vals[0],vals[1],time(eg)])
   for g,out,m in zip(gs,outs,merges):g.replay();torch.cuda.synchronize();validate(data,out,m)
   empty=statistics.median(r[2] for r in raw)
   rec=dict(shape=c,mode=mode,full_us=statistics.median(r[0] for r in raw)-empty,diagnostic_us=statistics.median(r[1] for r in raw)-empty,paired_delta_us=statistics.median(r[1]-r[0] for r in raw),splits=merges[0].args[1].shape[2],raw=raw)
   results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
   del gs,outs,g,out,merges,m;torch.cuda.empty_cache()
  del data;torch.cuda.empty_cache()
  (ROOT/f'results/diagnostic-{a.seed}.json').write_text(json.dumps(results,indent=2)+'\n')
